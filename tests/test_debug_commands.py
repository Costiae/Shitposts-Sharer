"""Debug command tests: /shit debug, /shit tags, /shit forceout|forcein."""

from __future__ import annotations

import asyncio
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.commands import HELP_ADMIN, HELP_ALL, HELP_TEXT, Commands  # noqa: E402
from core.config import DEFAULTS, Settings  # noqa: E402
from core.models import Direction, GroupProfile, Message, Segment  # noqa: E402
from core.pipeline import Loggerish, Pipeline  # noqa: E402
from core.storage import Storage  # noqa: E402
from tests.test_command_helpers import FakeEvent  # noqa: E402
from tests.test_core import scratch_path  # noqa: E402

GROUP_A = "aiocqhttp:GroupMessage:100001"
GROUP_B = "aiocqhttp:GroupMessage:200002"


class FakeContext:
    def __init__(self):
        self.sent = []

    async def send_message(self, group_key, chain):
        self.sent.append((group_key, chain))


class QuietLogger(Loggerish):
    def __init__(self):
        super().__init__(None)

    def _log(self, level, text):
        pass


class TestDebugCommands(unittest.TestCase):
    def setUp(self):
        import core.sender as sender_module
        from tests.test_pipeline import _stub_plain

        self._orig_plain = sender_module._plain
        sender_module._plain = _stub_plain

        self.db = scratch_path("debug")
        self.storage = Storage(self.db)
        self.settings = Settings(dict(DEFAULTS))
        self.context = FakeContext()
        self.pipeline = Pipeline(
            context=self.context,
            settings=self.settings,
            storage=self.storage,
            logger=QuietLogger(),
        )
        self.commands = Commands(
            self.storage,
            self.settings,
            probe_llm=self.pipeline.llm,
            probe_pipeline=self.pipeline,
            probe_force=self.pipeline,
        )
        self.storage.upsert_group(
            GroupProfile(group_key=GROUP_A, name="A", direction=Direction.EMIT)
        )
        self.storage.upsert_group(
            GroupProfile(group_key=GROUP_B, name="B", direction=Direction.RECEIVE)
        )
        self.event = FakeEvent(GROUP_A, "A")

    def tearDown(self):
        import core.sender as sender_module

        sender_module._plain = self._orig_plain
        self.storage.close()
        for suffix in ("", "-wal", "-shm"):
            Path(str(self.db) + suffix).unlink(missing_ok=True)

    # ---------------------------------------------------------------- debug
    def test_debug_lists_gate_status(self):
        result = self.commands.dispatch(self.event, ["debug"], is_admin=True)
        self.assertTrue(result.ok, result.text)
        self.assertIn("PASS", result.text)
        self.assertIn("can emit (direction)", result.text)
        # A 群是 out，不该能收
        self.assertIn("[FAIL] can receive (direction)", result.text)

    def test_debug_is_admin_only(self):
        self.assertFalse(self.commands.dispatch(self.event, ["debug"]).ok)

    # ---------------------------------------------------------------- tags
    def test_tags_empty_then_populated(self):
        empty = self.commands.dispatch(self.event, ["tags"], is_admin=True)
        self.assertIn("top: -", empty.text)  # 汇总视图仍然列出群，只是没有词

        group = self.storage.get_group(GROUP_B)
        group.intake = {"abstr": 0.5, "meme": 0.3}
        self.storage.upsert_group(group)

        listing = self.commands.dispatch(self.event, ["tags"], is_admin=True)
        self.assertIn("abstr", listing.text)

        detail = self.commands.dispatch(self.event, ["tags", GROUP_B], is_admin=True)
        self.assertTrue(detail.ok)
        self.assertIn("0.5000", detail.text)

    def test_tags_is_admin_only(self):
        self.assertFalse(self.commands.dispatch(self.event, ["tags"]).ok)

    # ---------------------------------------------------------------- force
    def _register_all_groups(self):
        # Pipeline is a slots dataclass: monkeypatching methods is not allowed.
        # Build the event the way AstrBot would instead.
        self.event.message_obj = type(
            "Obj",
            (),
            {
                "group_name": "A",
                "message": [Segment(kind="plain", text="强制搬运的测试内容")],
                "sender_name": "admin",
                "message_id": "m1",
                "time": 0,
            },
        )()
        self.event.get_sender_id = lambda: "admin1"  # type: ignore[assignment]
        # No card components here, so the real _enrich is a no-op.

    def test_forceout_delivers_ignoring_direction(self):
        self._register_all_groups()
        result = self.commands.dispatch(self.event, ["relay"], is_admin=True)
        self.assertTrue(result.ok, result.text)
        self.assertIn(GROUP_B, result.text)
        sent_groups = [key for key, _ in self.context.sent]
        self.assertIn(GROUP_B, sent_groups)
        # 不指定 target 时，不能把屎原样回声给源群
        self.assertNotIn(GROUP_A, sent_groups)

    def test_forceout_respects_dry_run(self):
        self._register_all_groups()
        self.settings.raw["dry_run"] = True
        result = self.commands.dispatch(self.event, ["relay"], is_admin=True)
        self.assertTrue(result.ok, result.text)
        self.assertIn("DRY RUN", result.text)
        self.assertEqual(self.context.sent, [])

    def test_forceout_image_requires_safety_check(self):
        self._register_all_groups()
        self.event.message_obj.message = [
            type("Img", (), {"type": "image", "url": "http://x/forced.png"})()
        ]
        result = self.commands.dispatch(self.event, ["relay"], is_admin=True)
        self.assertFalse(result.ok)
        self.assertIn("safety check unavailable", result.text)
        self.assertEqual(self.context.sent, [])

    def test_forceout_with_target(self):
        self._register_all_groups()
        result = self.commands.dispatch(
            self.event, ["relay", "200002"], is_admin=True
        )
        self.assertTrue(result.ok, result.text)
        self.assertEqual([key for key, _ in self.context.sent], [GROUP_B])

    def test_forcein_updates_intake_without_sending(self):
        self._register_all_groups()
        result = self.commands.dispatch(
            self.event, ["intake", "200002"], is_admin=True
        )
        self.assertTrue(result.ok, result.text)
        self.assertEqual(self.context.sent, [])
        # 口味被更新了（哪怕只是分词出来的弱向量）
        self.assertTrue(self.storage.get_group(GROUP_B).intake)

    def test_force_is_admin_only(self):
        denied = self.commands.dispatch(self.event, ["relay"])
        self.assertFalse(denied.ok)
        self.assertEqual(self.context.sent, [])

    # ---------------------------------------------------------------- tidy
    def test_tidy_targets_group_and_requires_admin(self):
        class Runner:
            def __init__(self):
                self.calls = []

            def run_tidy(self, key):
                self.calls.append(key)
                return f"tidy {key}"

        runner = Runner()
        self.commands.probe_force = runner

        denied = self.commands.dispatch(self.event, ["tidy"])
        self.assertFalse(denied.ok)
        self.assertEqual(runner.calls, [])

        result = self.commands.dispatch(
            self.event, ["tidy", "200002"], is_admin=True
        )
        self.assertTrue(result.ok, result.text)
        self.assertIn(GROUP_B, result.text)
        self.assertEqual(runner.calls, [GROUP_B])

    def test_tidy_rewrites_stored_taste(self):
        group = self.storage.get_group(GROUP_B)
        group.intake = {
            "zz": 0.2,
            "robot": 0.2,
            "robots": 0.2,
            "ai": 0.2,
            "code": 0.2,
        }
        self.storage.upsert_group(group)

        class FakeAnalyzer:
            def __init__(self):
                self.seen = []

            async def tidy_terms(self, terms, umo=""):
                self.seen = list(terms)
                return {
                    "drop": ["zz"],
                    "groups": [
                        {"canonical": "人工智能", "members": ["ai"]},
                        {"canonical": "编程", "members": ["code"]},
                    ],
                }

        analyzer = FakeAnalyzer()
        self.pipeline.analyzer = analyzer
        result = self.commands.dispatch(
            self.event, ["tidy", "200002"], is_admin=True
        )
        self.assertTrue(result.ok, result.text)

        intake = self.storage.get_group(GROUP_B).intake
        self.assertNotIn("zz", intake)
        self.assertIn("robots", intake)  # robot 已并进 robots
        self.assertIn("人工智能", intake)
        self.assertIn("编程", intake)
        self.assertTrue(any(term == "robots" for term, _ in analyzer.seen))

    # ---------------------------------------------------------------- help
    def test_help_mentions_debug_commands(self):
        for token in ("debug", "tags", "tidy", "relay", "intake"):
            self.assertIn(token, HELP_ADMIN)
        for block in (HELP_TEXT, HELP_ADMIN, HELP_ALL):
            block.encode("ascii")

    def test_help_states_direction_subject_is_the_group(self):
        """帮助里必须写清主语：direction 描述的是群，不是插件。"""
        self.assertIn("THE GROUP, not the plugin", HELP_TEXT)
        self.assertIn("receive", HELP_TEXT)
        self.assertIn("emit", HELP_TEXT)

    def test_help_rejects_no_stale_in_out_wording(self):
        stale = "in = receive only, out = emit only"
        self.assertNotIn(stale, HELP_TEXT)


if __name__ == "__main__":
    unittest.main()
