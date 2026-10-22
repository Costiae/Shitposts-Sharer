"""端到端：假 AstrBot 上下文 + 假 LLM，跑通“读消息 → 判屎 → 搬运”。

这里不 import AstrBot，用一个 stand-in context 验证流程编排本身。
"""

from __future__ import annotations

import asyncio
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core import sender as sender_module  # noqa: E402
from core.config import DEFAULTS, Settings  # noqa: E402
from core.models import Direction, Message, Segment  # noqa: E402
from core.pipeline import Loggerish, Pipeline  # noqa: E402
from core.storage import Storage  # noqa: E402
from tests.test_core import scratch_path  # noqa: E402


class StubComponent:
    """代替 AstrBot 的 Plain。

    本机没有 astrbot 包，真组件构造不了；但链路测试只关心“组件顺序与内容”，
    所以用最小替身。sender 已经全部走模块级 _plain，补丁一处就够。
    """

    __test__ = False

    def __init__(self, text: str) -> None:
        self.text = text

    def __repr__(self) -> str:  # 便于断言失败时看清楚
        return f"Plain({self.text!r})"


def _stub_plain(text: str):
    return StubComponent(text)


class FakeResponse:
    def __init__(self, text: str) -> None:
        self.completion_text = text


class FakeContext:
    """只实现 llm_generate 与 send_message。

    必须提供 get_using_provider_async：AstrBot 的 llm_generate 要求
    chat_provider_id 必填，LLMClient 在配置留空时会来解析默认模型；
    解析不到就不会发起调用（见 tests/test_llm.py）。
    """

    def __init__(self, replies: list[str]) -> None:
        self.replies = list(replies)
        self.sent: list[tuple[str, list]] = []
        self.prompts: list[str] = []

    async def get_using_provider_async(self, umo=None, *, locale="zh-CN"):
        return type(
            "P", (), {"provider_config": {"id": "fake-provider"}}
        )()

    async def llm_generate(self, **kwargs):
        self.prompts.append(kwargs.get("prompt", ""))
        text = self.replies.pop(0) if self.replies else "{}"
        return FakeResponse(text)

    async def send_message(self, group_key, chain):
        self.sent.append((group_key, chain))


class QuietLogger(Loggerish):
    def __init__(self) -> None:
        super().__init__(None)
        self.lines: list[str] = []

    def _log(self, level, text):
        self.lines.append(f"{level}:{text}")


class TestPipeline(unittest.TestCase):
    def setUp(self) -> None:
        self.settings = Settings(dict(DEFAULTS))
        self.settings.raw["llm_rerank"] = True
        self.db = scratch_path("pipeline")
        self.storage = Storage(self.db)
        self._original_plain = sender_module._plain
        self._original_image = sender_module._image
        sender_module._plain = _stub_plain
        sender_module._image = lambda url: StubComponent(f"Image({url})")

    def tearDown(self) -> None:
        sender_module._plain = self._original_plain
        sender_module._image = self._original_image
        self.storage.close()
        for suffix in ("", "-wal", "-shm"):
            Path(str(self.db) + suffix).unlink(missing_ok=True)

    def _setup_groups(self):
        from core.vector import build_vector

        source = self.storage.get_group("src") or None
        from core.models import GroupProfile

        self.storage.upsert_group(
            GroupProfile(group_key="src", name="出屎群", direction=Direction.EMIT)
        )
        self.storage.upsert_group(
            GroupProfile(
                group_key="sink",
                name="吃屎甲",
                direction=Direction.RECEIVE,
                intake=build_vector(["抽象", "离谱"], "抽象离谱的玩梗内容", 0.9),
                native_message_count=40,
                native_char_count=2000,
            )
        )
        self.storage.upsert_group(
            GroupProfile(
                group_key="other",
                name="正经群",
                direction=Direction.RECEIVE,
                intake=build_vector(["考研", "论文"], "考研论文进度", 0.2),
                native_message_count=40,
                native_char_count=2000,
            )
        )

    def test_full_flow(self):
        self._setup_groups()

        judge = (
            '{"is_shit": true, "intensity": 0.92, "tags": ["离谱"],'
            ' "keywords": ["抽象", "离谱", "逆天"], "reason": "太抽象了",'
            ' "context_relevant": true, "context_keywords": ["玩梗"],'
            ' "topic_seed": "群友在比谁更抽象"}'
        )
        rerank = '{"scores": {"sink": 0.95, "other": 0.05}, "reason": "甲群爱这口"}'

        context = FakeContext([judge, rerank])
        pipeline = Pipeline(
            context=context,
            settings=self.settings,
            storage=self.storage,
            logger=QuietLogger(),
        )

        message = Message(
            group_key="src",
            sender_name="甲",
            segments=[
                Segment(
                    kind="plain",
                    text="我昨天梦见用抽象数轴证明离谱猜想，然后被导师当场开除，太逆天了",
                )
            ],
        )
        suspicion = pipeline.accumulator.open(message, "长文本")

        result = asyncio.run(pipeline.handle_suspicion(suspicion))
        self.assertIsNotNone(result)
        self.assertEqual(result["delivered"], ["sink"])

        self.assertEqual(len(context.sent), 1)
        target, chain = context.sent[0]
        self.assertEqual(target, "sink")
        self.assertTrue(chain)

        # 来源群记录了口味与样本
        source = self.storage.get_group("src")
        self.assertEqual(source.native_message_count, 1)
        self.assertTrue(source.intake)

        # 重复的屎不会再搬一次
        again = pipeline.accumulator.open(message, "长文本")
        context.replies = [judge, rerank]
        dup = asyncio.run(pipeline.handle_suspicion(again))
        self.assertTrue(dup.get("duplicate"))

    def test_source_group_not_duplicated_when_in_both(self):
        self._setup_groups()
        group = self.storage.get_group("src")
        group.direction = Direction.BOTH
        self.storage.upsert_group(group)

        judge = (
            '{"is_shit": true, "intensity": 0.9, "keywords": ["抽象"],'
            ' "context_relevant": false}'
        )
        context = FakeContext([judge, '{"scores": {"sink": 0.9}}'])
        pipeline = Pipeline(
            context=context,
            settings=self.settings,
            storage=self.storage,
            logger=QuietLogger(),
        )
        message = Message(
            group_key="src",
            segments=[Segment(kind="plain", text="抽象到离谱的发言内容，笑着活下去")],
        )
        result = asyncio.run(pipeline.handle_suspicion(pipeline.accumulator.open(message, "长文本")))
        self.assertIn("sink", result["delivered"])
        self.assertNotIn("src", result["delivered"])

    def test_image_delivered_only_to_group_matching_visual_tags(self):
        from core.vector import build_vector

        self._setup_groups()
        self.settings.raw["llm_rerank"] = False
        sink = self.storage.get_group("sink")
        sink.intake = build_vector(["初音未来", "白丝"], "", 0.9)
        self.storage.upsert_group(sink)

        judge = (
            '{"is_shit": true, "intensity": 0.9, "tags": ["好看"],'
            ' "keywords": ["初音未来"], "image_tags": ["初音未来", "白丝"],'
            ' "characters": ["初音未来"], "copyrights": ["VOCALOID"],'
            ' "nsfw": true, "minor_sexual": false, "reason": "对味"}'
        )
        context = FakeContext([judge])
        pipeline = Pipeline(
            context=context,
            settings=self.settings,
            storage=self.storage,
            logger=QuietLogger(),
        )
        message = Message(
            group_key="src",
            segments=[Segment(kind="image", url="http://x/cute.png")],
        )
        result = asyncio.run(
            pipeline.handle_suspicion(pipeline.accumulator.open(message, "媒体"))
        )

        self.assertIsNotNone(result)
        self.assertEqual(result["delivered"], ["sink"])
        self.assertIn("图片标签", dict(result["rejected"])["other"])

        record = self.storage.get_shit(result["shit_id"])
        self.assertEqual(record.kind.value, "image")
        self.assertIn("初音未来", record.image_tags)
        self.assertIn("NSFW", record.image_tags)
        self.assertTrue(record.nsfw)
        self.assertGreater(max(record.image_vector.values()), 0.0)

    def test_image_disabled_source_group_is_skipped_before_judge(self):
        self._setup_groups()
        source = self.storage.get_group("src")
        source.image_enabled = False
        self.storage.upsert_group(source)

        context = FakeContext([])
        pipeline = Pipeline(
            context=context,
            settings=self.settings,
            storage=self.storage,
            logger=QuietLogger(),
        )
        message = Message(
            group_key="src",
            segments=[Segment(kind="image", url="http://x/ignored.png")],
        )
        result = asyncio.run(
            pipeline.handle_suspicion(pipeline.accumulator.open(message, "媒体"))
        )

        self.assertIsNone(result)
        self.assertEqual(context.prompts, [])
        self.assertEqual(pipeline.stats.judged, 0)
        self.assertEqual(context.sent, [])

    def test_minor_sexual_content_is_hard_blocked(self):
        self._setup_groups()
        self.settings.raw["llm_rerank"] = False

        judge = (
            '{"is_shit": true, "intensity": 0.95, "image_tags": ["测试标签"],'
            ' "nsfw": true, "minor_sexual": true, "reason": "安全拦截"}'
        )
        context = FakeContext([judge])
        pipeline = Pipeline(
            context=context,
            settings=self.settings,
            storage=self.storage,
            logger=QuietLogger(),
        )
        message = Message(
            group_key="src",
            segments=[Segment(kind="image", url="http://x/blocked.png")],
        )
        result = asyncio.run(
            pipeline.handle_suspicion(pipeline.accumulator.open(message, "媒体"))
        )

        self.assertIsNone(result)
        self.assertEqual(context.sent, [])
        self.assertEqual(pipeline.stats.blocked_safety, 1)
        # 不进口味、不入库，避免以后被重新路由出去
        self.assertEqual(self.storage.get_group("src").native_message_count, 0)


if __name__ == "__main__":
    unittest.main()
