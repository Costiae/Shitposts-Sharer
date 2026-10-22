"""核心逻辑单元测试（不依赖 AstrBot 运行时，只用标准库 unittest）。

运行：``python -m unittest discover -s tests -v``
"""

from __future__ import annotations

import itertools
import sys
import time
import unittest
import uuid
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.accumulator import ContextAccumulator  # noqa: E402
from core.bili import extract_bili_ids  # noqa: E402
from core.config import DEFAULTS, Settings  # noqa: E402
from core.detector import is_there_a_shit  # noqa: E402
from core.llm import extract_json  # noqa: E402
from core.models import (  # noqa: E402
    Direction,
    GroupProfile,
    Kind,
    Message,
    Segment,
    ShitRecord,
)
from core.profile import confidence_score, update_intake  # noqa: E402
from core.router import Router, fingerprint_of  # noqa: E402
from core.storage import Storage  # noqa: E402
from core.vector import build_vector, cosine, keyword_score  # noqa: E402


def make_settings(**overrides) -> Settings:
    raw = dict(DEFAULTS)
    raw.update(overrides)
    return Settings(raw)


def make_group(
    key: str, direction: Direction = Direction.BOTH, native: int = 20
) -> GroupProfile:
    return GroupProfile(
        group_key=key,
        name=key,
        direction=direction,
        intake=build_vector(["抽象", "离谱"], "抽象离谱的玩梗内容", 0.8),
        native_message_count=native,
        native_char_count=native * 50,
    )


def make_shit(text: str, keywords: list[str], group_key: str = "g-source") -> ShitRecord:
    record = ShitRecord(
        shit_id="s1",
        source_group=group_key,
        kind=Kind.TEXT,
        text_blob=text,
        keywords=keywords,
        vector=build_vector(keywords, text, 0.8),
    )
    record.fingerprint = fingerprint_of(record)
    return record


class TestDirection(unittest.TestCase):
    def test_parse_and_capabilities(self):
        self.assertIs(Direction.parse("both"), Direction.BOTH)
        self.assertIs(Direction.parse("all"), Direction.BOTH)
        self.assertIs(Direction.parse("recv"), Direction.RECEIVE)
        self.assertTrue(Direction.RECEIVE.can_receive)
        self.assertFalse(Direction.RECEIVE.can_emit)
        self.assertTrue(Direction.EMIT.can_emit)
        self.assertFalse(Direction.EMIT.can_receive)
        self.assertTrue(Direction.BOTH.can_emit and Direction.BOTH.can_receive)
        with self.assertRaises(ValueError):
            Direction.parse("bogus")

    def test_old_and_chinese_names_are_gone(self):
        """语言统一：in/out 与中文别名都不再被 parse 接受。"""
        for token in ("in", "out", "收", "发", "进出"):
            with self.assertRaises(ValueError, msg=f"{token} should be rejected"):
                Direction.parse(token)

    def test_values_align_with_capability_names(self):
        """字段值与 can_receive/can_emit 对齐，避免 in/out 那种主语歧义。"""
        self.assertEqual(Direction.RECEIVE.value, "receive")
        self.assertEqual(Direction.EMIT.value, "emit")
        self.assertEqual(Direction.BOTH.value, "both")

    def test_describe_names_the_group_as_subject(self):
        for direction in Direction:
            self.assertIn("this group", direction.describe())


class TestVector(unittest.TestCase):
    def test_similarity_prefers_shared_topic(self):
        a = build_vector(["抽象", "玩梗"], "抽象玩梗高手", 0.8)
        b = build_vector(["抽象", "玩梗"], "抽象玩梗低手", 0.7)
        c = build_vector(["考研", "数学"], "考研数学加油", 0.1)
        self.assertGreater(cosine(a, b), cosine(a, c))
        self.assertAlmostEqual(cosine(a, a), 1.0, places=6)

    def test_tolerance_band(self):
        # 相近：共享主题词，余弦接近但略低于阈值 → 给及格分
        shit = build_vector(["抽象", "离谱", "逆天"], "抽象离谱的逆天发言内容", 0.5)
        close = build_vector(["抽象", "离谱", "玩梗"], "抽象离谱的玩梗内容", 0.5)
        far = build_vector(["考研", "数学"], "考研数学加油", 0.5)

        similar = cosine(shit, close)
        self.assertTrue(0.10 < similar < 0.75, f"夹具偏离预期: {similar:.3f}")

        near = keyword_score(shit, close, threshold=0.75, tolerance=0.10)
        self.assertGreater(near, 0.0)
        # 相似度一旦掉到容差带以下，就不该再给分
        strict = keyword_score(shit, close, threshold=similar + 0.20, tolerance=0.05)
        self.assertEqual(strict, 0.0)
        self.assertEqual(keyword_score(shit, far, threshold=0.75), 0.0)


class TestConfidence(unittest.TestCase):
    def test_grows_with_samples_and_falls_with_rejections(self):
        low = confidence_score(
            native_messages=1, native_chars=40, drift=0, accepted=1, rejected=0
        )
        high = confidence_score(
            native_messages=80, native_chars=3200, drift=0, accepted=160, rejected=0
        )
        rejected = confidence_score(
            native_messages=80, native_chars=3200, drift=0, accepted=1, rejected=200
        )
        self.assertLess(low, 0.2)
        self.assertGreater(high, 0.7)
        self.assertLess(rejected, high)

    def test_update_intake_accumulates(self):
        settings = make_settings()
        group = make_group("g1", native=0)
        before = dict(group.intake)
        shit = make_shit("抽象离谱，逆天发言", ["抽象", "逆天"])
        drift = update_intake(group, shit, settings, is_native=True)
        self.assertGreaterEqual(drift, 0.0)
        self.assertLessEqual(drift, 1.0)
        self.assertEqual(group.native_message_count, 1)
        self.assertTrue(set(group.intake) >= set(before) | {"逆天"})


class TestDetector(unittest.TestCase):
    def setUp(self):
        self.settings = make_settings()

    def test_forward_image_bili(self):
        forward = Message(
            group_key="g",
            segments=[
                Segment(
                    kind="forward", children=[Segment(kind="plain", text="哈哈哈")]
                )
            ],
        )
        self.assertTrue(is_there_a_shit(forward, self.settings)[0])

        image = Message(
            group_key="g", segments=[Segment(kind="image", url="http://x/y.png")]
        )
        self.assertTrue(is_there_a_shit(image, self.settings)[0])

        bili = Message(
            group_key="g", segments=[Segment(kind="plain", text="看这个 BV1xx411c7mD")]
        )
        self.assertTrue(is_there_a_shit(bili, self.settings)[0])

    def test_short_chatter_ignored(self):
        short = Message(group_key="g", segments=[Segment(kind="plain", text="哈哈")])
        self.assertEqual(is_there_a_shit(short, self.settings), (False, ""))

    def test_bili_extraction(self):
        refs = extract_bili_ids("看 https://www.bilibili.com/video/BV1xx411c7mD 和 BV1xx411c7mD")
        self.assertEqual(refs.bvids, ["BV1xx411c7mD"])


class TestRouter(unittest.TestCase):
    def setUp(self):
        # 全局阈值抬到 0.6，让“可信度”这道门真正起作用
        self.settings = make_settings(threshold_overall=0.6)
        self.router = Router(self.settings)
        self.shit = make_shit("抽象离谱到爆的内容", ["抽象", "离谱"])

    def test_direction_and_threshold(self):
        groups = [
            make_group("out-only", Direction.EMIT),
            make_group("in-only", Direction.RECEIVE, native=50),
            make_group("too-fresh", Direction.RECEIVE, native=0),
        ]
        plan = self.router.keyword_plan(self.shit, groups, exclude={"g-source"})
        keys = plan.group_keys()
        self.assertNotIn("out-only", keys)  # 只发的群不收货
        self.assertIn("in-only", keys)  # 口味对、样本足
        self.assertNotIn("too-fresh", keys)  # 口味对但没吃过屎，可信度不足

        reasons = dict(plan.rejected)
        self.assertIn("可信度不足", reasons["too-fresh"])

    def test_low_confidence_does_not_block_when_threshold_is_zero(self):
        settings = make_settings(threshold_overall=0.0)
        router = Router(settings)
        plan = router.keyword_plan(
            self.shit, [make_group("fresh", Direction.RECEIVE, native=0)]
        )
        # 新鲜群也能收（默认阈值 0 只防“口味不像”，不防“样本少”）
        self.assertIn("fresh", plan.group_keys())

    def test_no_repeat_delivery(self):
        groups = [make_group("in-only", Direction.RECEIVE)]
        plan = self.router.keyword_plan(self.shit, groups, delivered={"in-only"})
        self.assertEqual(plan.candidates, [])
        self.assertTrue(plan.rejected)


_WORKDIR = ROOT / "tests" / "_scratch"
_seq = itertools.count()


def scratch_path(name: str) -> Path:
    """在仓库内开一个隔离的数据库路径。

    不用 tempfile/TemporaryDirectory：Windows 沙箱会接管它创建的目录，
    进程随即失去 WRITE_DAC，连读都读不了，更没法清理。直接 makedirs 就行。
    """
    folder = _WORKDIR / f"{name}-{next(_seq)}-{uuid.uuid4().hex[:8]}"
    folder.mkdir(parents=True, exist_ok=True)
    return folder / "t.sqlite3"


class TestStorage(unittest.TestCase):
    def setUp(self):
        self.db_path = scratch_path("storage")
        self.storage = Storage(self.db_path)

    def tearDown(self):
        self.storage.close()
        for suffix in ("", "-wal", "-shm"):
            Path(str(self.db_path) + suffix).unlink(missing_ok=True)

    def test_roundtrip_and_delivery_dedup(self):
        group = make_group("g1")
        self.storage.upsert_group(group)
        self.assertIs(self.storage.get_group("g1").direction, Direction.BOTH)

        shit = make_shit("抽象离谱", ["抽象", "离谱"])
        self.storage.add_shit(shit)
        self.assertEqual(
            self.storage.find_by_fingerprint(shit.fingerprint).shit_id, shit.shit_id
        )

        self.assertTrue(
            self.storage.record_delivery(shit.shit_id, "g1", 0.9, 0.8, "keyword")
        )
        self.assertFalse(
            self.storage.record_delivery(shit.shit_id, "g1", 0.9, 0.8, "keyword")
        )
        self.assertTrue(self.storage.delete_group("g1"))
        self.assertIsNone(self.storage.get_group("g1"))


class TestJsonParsing(unittest.TestCase):
    def test_extract_json(self):
        self.assertEqual(extract_json('```json\n{"a": 1}\n```'), {"a": 1})
        self.assertEqual(
            extract_json('好的，结果如下：{"is_shit": true, "intensity": 0.9} 谢谢'),
            {"is_shit": True, "intensity": 0.9},
        )
        self.assertIsNone(extract_json("没有 JSON"))


class TestAccumulator(unittest.TestCase):
    def test_window_close(self):
        settings = make_settings(window_seconds=10)
        acc = ContextAccumulator(settings)
        msg = Message(group_key="g", segments=[Segment(kind="plain", text="抽象大屎")])
        acc.observe(msg)
        acc.open(msg, "长文本")
        self.assertEqual(acc.due(now=time.time()), [])

        ready = acc.due(now=time.time() + 60)
        self.assertEqual(len(ready), 1)
        self.assertEqual(ready[0][0], "g")


if __name__ == "__main__":
    unittest.main()
