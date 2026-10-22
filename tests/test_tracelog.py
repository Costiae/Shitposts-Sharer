"""Trace-logging tests.

Requirement: every group message must be accounted for in the log, with
per-layer verdicts - caught or not, and why not - plus the analysis input
and the model output.
"""

from __future__ import annotations

import asyncio
import sys
import time
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.config import DEFAULTS, Settings  # noqa: E402
from core.models import Direction, GroupProfile, Message, Segment  # noqa: E402
from core.pipeline import Loggerish, Pipeline  # noqa: E402
from core.storage import Storage  # noqa: E402
from core.tracelog import Trace, describe_message, describe_segments  # noqa: E402
from tests.test_core import scratch_path  # noqa: E402

GROUP = "aiocqhttp:GroupMessage:100001"


class CapturingLogger(Loggerish):
    def __init__(self):
        super().__init__(None)
        self.lines: list[str] = []

    def _log(self, level, text):
        self.lines.append(f"{level}|{text}")

    def text(self) -> str:
        return "\n".join(self.lines)


class FakeContext:
    def __init__(self, reply: str = '{"is_shit": false}'):
        self.reply = reply
        self.sent: list[tuple[str, list]] = []
        self.prompts: list[str] = []

    async def get_using_provider_async(self, umo=None, *, locale="zh-CN"):
        return type("P", (), {"provider_config": {"id": "p"}})()

    async def llm_generate(self, *, chat_provider_id, prompt=None, system_prompt=None, **kw):
        self.prompts.append(prompt or "")
        return type("R", (), {"completion_text": self.reply})()

    async def send_message(self, group_key, chain):
        self.sent.append((group_key, chain))


class FakeEvent:
    def __init__(self, text: str, origin: str = GROUP, name: str = "群友"):
        self.unified_msg_origin = origin
        self.message_str = text
        self.message_obj = type(
            "Obj",
            (),
            {
                "group_name": "测试群",
                "message": [Segment(kind="plain", text=text)],
                "sender_name": name,
                "message_id": "m1",
                "time": 0,
            },
        )()

    def get_sender_id(self):
        return "1001"


class TestTraceLevels(unittest.TestCase):
    def setUp(self):
        self.logger = CapturingLogger()

    def test_off_writes_nothing_for_steps(self):
        trace = Trace(self.logger, "off")
        trace.gate("x")
        trace.step("should not appear")
        trace.skip("should not appear either")
        self.assertEqual(self.logger.lines, [])

    def test_normal_writes_caught_but_not_steps(self):
        trace = Trace(self.logger, "normal")
        trace.gate("detect")
        trace.step("hidden")
        trace.caught("long text")
        text = self.logger.text()
        self.assertIn("CAUGHT long text", text)
        self.assertNotIn("hidden", text)

    def test_normal_writes_skip_reason(self):
        trace = Trace(self.logger, "normal")
        trace.gate("detect")
        trace.skip("too short")
        self.assertIn("SKIP too short", self.logger.text())

    def test_full_writes_steps_and_skips(self):
        trace = Trace(self.logger, "full")
        trace.gate("detect")
        trace.step("visible step")
        trace.skip("too short")
        text = self.logger.text()
        self.assertIn("visible step", text)
        self.assertIn("SKIP too short", text)

    def test_skip_reason_is_always_kept_in_trail(self):
        trace = Trace(self.logger, "normal")
        trace.gate("group")
        trace.skip("not registered")
        self.assertIn("group: not registered", trace.summary())

    def test_unknown_level_falls_back_to_normal(self):
        self.assertEqual(Trace(self.logger, "bogus").level, "normal")

    def test_all_lines_are_prefixed(self):
        trace = Trace(self.logger, "full")
        trace.gate("x")
        trace.step("a")
        trace.caught("b")
        for line in self.logger.lines:
            self.assertIn("[shit]", line)


class TestDescriptions(unittest.TestCase):
    def test_describe_message_includes_kinds_and_text(self):
        message = Message(
            group_key=GROUP,
            sender_name="甲",
            segments=[Segment(kind="plain", text="你好呀")],
            message_id="m9",
        )
        described = describe_message(message)
        self.assertIn("kinds=plain", described)
        self.assertIn("你好呀", described)
        self.assertIn("m9", described)

    def test_describe_message_truncates(self):
        message = Message(
            group_key=GROUP,
            segments=[Segment(kind="plain", text="字" * 500)],
        )
        self.assertLess(len(describe_message(message, limit=100)), 160)

    def test_describe_segments_enumerates(self):
        message = Message(
            group_key=GROUP,
            segments=[
                Segment(kind="plain", text="文字"),
                Segment(kind="image", url="http://x/y.png"),
                Segment(
                    kind="forward",
                    url="42",
                    children=[Segment(kind="plain", text="展开内容")],
                ),
            ],
        )
        described = describe_segments(message)
        self.assertIn("#0:plain", described)
        self.assertIn("#1:image", described)
        self.assertIn("#2:forward", described)
        self.assertIn("1 children", described)
        # 转发内部内容也要展开，否则日志看不出「转发里到底有啥」
        self.assertIn("展开内容", described)

    def test_describe_segments_marks_hollow_forward(self):
        message = Message(
            group_key=GROUP,
            segments=[Segment(kind="forward", url="42")],
        )
        described = describe_segments(message)
        self.assertIn("forward", described)
        self.assertNotIn("children", described)

    def test_hollow_forward_text_is_explicit(self):
        message = Message(group_key=GROUP, segments=[Segment(kind="forward", url="42")])
        self.assertIn("未展开", message.text())


class TestPipelineLogging(unittest.TestCase):
    def setUp(self):
        import core.sender as sender_module
        from tests.test_pipeline import _stub_plain

        self._orig_plain = sender_module._plain
        sender_module._plain = _stub_plain

        self.db = scratch_path("trace")
        self.storage = Storage(self.db)

    def tearDown(self):
        import core.sender as sender_module

        sender_module._plain = self._orig_plain
        self.storage.close()
        for suffix in ("", "-wal", "-shm"):
            Path(str(self.db) + suffix).unlink(missing_ok=True)

    def _pipeline(self, reply: str, level: str = "full"):
        raw = dict(DEFAULTS)
        raw["log_level"] = level
        raw["trigger_cooldown_sec"] = 0
        settings = Settings(raw)
        logger = CapturingLogger()
        context = FakeContext(reply)
        pipeline = Pipeline(
            context=context, settings=settings, storage=self.storage, logger=logger
        )
        return pipeline, logger, context

    def _register(self, direction=Direction.EMIT):
        self.storage.upsert_group(
            GroupProfile(group_key=GROUP, name="测试群", direction=direction)
        )

    def test_unregistered_group_is_logged_as_skip(self):
        pipeline, logger, _ = self._pipeline('{"is_shit": false}')
        asyncio.run(pipeline.on_event(FakeEvent("随便聊聊")))
        text = logger.text()
        self.assertIn("SKIP not registered", text)

    def test_paused_group_is_logged(self):
        self._register()
        group = self.storage.get_group(GROUP)
        group.enabled = False
        self.storage.upsert_group(group)

        pipeline, logger, _ = self._pipeline('{"is_shit": false}')
        asyncio.run(pipeline.on_event(FakeEvent("随便聊聊")))
        self.assertIn("SKIP group paused", logger.text())

    def test_receive_only_group_logs_context_only(self):
        self._register(Direction.RECEIVE)
        pipeline, logger, _ = self._pipeline('{"is_shit": false}')
        asyncio.run(pipeline.on_event(FakeEvent("随便聊聊")))
        self.assertIn("receive-only", logger.text())

    def test_short_message_logs_why_not_suspicious(self):
        self._register()
        pipeline, logger, _ = self._pipeline('{"is_shit": false}')
        asyncio.run(pipeline.on_event(FakeEvent("嗯")))
        text = logger.text()
        self.assertIn("SKIP not suspicious", text)
        self.assertIn("need=", text)

    def test_caught_message_logs_reason_and_content(self):
        self._register()
        long_text = "这是一段足够长的群聊内容" * 6
        pipeline, logger, _ = self._pipeline('{"is_shit": false}')
        asyncio.run(pipeline.on_event(FakeEvent(long_text)))
        text = logger.text()
        self.assertIn("CAUGHT", text)
        self.assertIn("长文本", text)

    def test_judge_logs_model_verdict_and_reason(self):
        self._register()
        long_text = "这是一段足够长的群聊内容" * 6
        pipeline, logger, context = self._pipeline(
            '{"is_shit": true, "intensity": 0.9, "reason": "很离谱"}'
        )
        event = FakeEvent(long_text)
        asyncio.run(pipeline.on_event(event))
        # 手动收盘（跳过等待窗口）
        results = asyncio.run(pipeline.flush_due(now=time.time() + 3600))
        self.assertTrue(results)
        text = logger.text()
        self.assertIn("window closed", text)
        self.assertIn("is_shit=True", text)
        self.assertIn("0.90", text)
        self.assertIn("很离谱", text)
        # 提示词确实发出了
        self.assertTrue(any("是否值得跨群搬运" in p for p in context.prompts))

    def test_low_intensity_is_logged_as_skip(self):
        self._register()
        long_text = "这是一段足够长的群聊内容" * 6
        pipeline, logger, _ = self._pipeline(
            '{"is_shit": true, "intensity": 0.2, "reason": "不够劲"}'
        )
        asyncio.run(pipeline.on_event(FakeEvent(long_text)))
        asyncio.run(pipeline.flush_due(now=time.time() + 3600))
        self.assertIn("intensity 0.20 < min", logger.text())

    def test_no_candidate_group_is_logged(self):
        self._register()
        long_text = "这是一段足够长的群聊内容" * 6
        pipeline, logger, _ = self._pipeline(
            '{"is_shit": true, "intensity": 0.9, "reason": "很离谱"}'
        )
        asyncio.run(pipeline.on_event(FakeEvent(long_text)))
        asyncio.run(pipeline.flush_due(now=time.time() + 3600))
        self.assertIn("no candidate group passed", logger.text())

    def test_off_level_is_quiet_on_skips(self):
        pipeline, logger, _ = self._pipeline('{"is_shit": false}', level="off")
        asyncio.run(pipeline.on_event(FakeEvent("嗯")))
        self.assertEqual(logger.lines, [])


if __name__ == "__main__":
    unittest.main()
