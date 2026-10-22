"""Command layer tests.

Covers both the in-group self-service path and the cross-group admin path
(operator sits in group A, toggles relay for group B).
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.commands import HELP_TEXT, CommandResult, Commands  # noqa: E402
from core.config import DEFAULTS, Settings  # noqa: E402
from core.models import Direction, GroupProfile  # noqa: E402
from core.storage import Storage  # noqa: E402
from core.targets import looks_like_target, resolve  # noqa: E402
from tests.test_core import scratch_path  # noqa: E402

GROUP_A = "aiocqhttp:GroupMessage:100001"
GROUP_B = "aiocqhttp:GroupMessage:200002"
GROUP_B_ID = "200002"


class FakeEvent:
    def __init__(self, origin: str = GROUP_A, name: str = "group A"):
        self.unified_msg_origin = origin
        self.message_obj = type("Obj", (), {"group_name": name})()

    def get_group_id(self):
        return self.unified_msg_origin.rsplit(":", 1)[-1]


class TestTargets(unittest.TestCase):
    def test_looks_like_target(self):
        self.assertTrue(looks_like_target("200002"))
        self.assertTrue(looks_like_target("#2"))
        self.assertTrue(looks_like_target("aiocqhttp:GroupMessage:200002"))
        self.assertFalse(looks_like_target("in"))
        self.assertFalse(looks_like_target("both"))
        self.assertFalse(looks_like_target("0.8"))

    def test_resolve_current_session(self):
        db = scratch_path("targets")
        storage = Storage(db)
        try:
            storage.upsert_group(
                GroupProfile(group_key=GROUP_A, name="group A", direction=Direction.RECEIVE)
            )
            event = FakeEvent()
            self.assertTrue(resolve(storage, event, "").ok)
            self.assertTrue(resolve(storage, event, GROUP_B_ID, want_unregistered=True).ok)
            self.assertFalse(resolve(storage, event, GROUP_B_ID).ok)
        finally:
            storage.close()
            for suffix in ("", "-wal", "-shm"):
                Path(str(db) + suffix).unlink(missing_ok=True)


class TestCommands(unittest.TestCase):
    def setUp(self):
        self.db = scratch_path("commands")
        self.storage = Storage(self.db)
        self.settings = Settings(dict(DEFAULTS))
        self.commands = Commands(self.storage, self.settings)
        self.event = FakeEvent(GROUP_A, "group A")

    def tearDown(self):
        self.storage.close()
        for suffix in ("", "-wal", "-shm"):
            Path(str(self.db) + suffix).unlink(missing_ok=True)

    # ---------------------------------------------------------------- ASCII
    def test_help_is_ascii(self):
        HELP_TEXT.encode("ascii")

    # ---------------------------------------------------------------- 自助
    def test_register_current_group(self):
        result = self.commands.dispatch(self.event, ["register", "both"])
        self.assertTrue(result.ok, result.text)
        group = self.storage.get_group(GROUP_A)
        self.assertIs(group.direction, Direction.BOTH)

    def test_register_direction_aliases(self):
        for token, expected in (
            ("receive", Direction.RECEIVE),
            ("recv", Direction.RECEIVE),
            ("emit", Direction.EMIT),
            ("both", Direction.BOTH),
            ("all", Direction.BOTH),
        ):
            self.commands.dispatch(self.event, ["register", token])
            self.assertIs(self.storage.get_group(GROUP_A).direction, expected)

    def test_register_rejects_old_in_out_names(self):
        """in/out 已废弃：语言统一，不再接受。"""
        for token in ("in", "out"):
            result = self.commands.dispatch(self.event, ["register", token])
            self.assertFalse(result.ok, f"{token} should be rejected")
            self.assertIn("receive", result.text)

    def test_register_rejects_chinese_aliases(self):
        result = self.commands.dispatch(self.event, ["register", "收"])
        self.assertFalse(result.ok)

    def test_status_and_list(self):
        self.assertEqual(
            self.commands.dispatch(self.event, ["list"]).text,
            "No group registered yet.",
        )
        self.commands.dispatch(self.event, ["register", "receive"])
        status = self.commands.dispatch(self.event, ["status"])
        self.assertIn("confidence", status.text)

        listing = self.commands.dispatch(self.event, ["list"])
        self.assertIn(GROUP_A, listing.text)
        self.assertIn("#1", listing.text)

    def test_unregister(self):
        self.commands.dispatch(self.event, ["register"])
        self.assertTrue(self.commands.dispatch(self.event, ["unregister"]).ok)
        self.assertIsNone(self.storage.get_group(GROUP_A))

    def test_threshold_percent(self):
        self.commands.dispatch(self.event, ["register", "receive", "80%"])
        self.assertAlmostEqual(self.storage.get_group(GROUP_A).threshold, 0.8, places=3)

    def test_bad_threshold_rejected(self):
        self.commands.dispatch(self.event, ["register"])
        self.assertFalse(
            self.commands.dispatch(self.event, ["threshold", "abc"]).ok
        )

    def test_unknown_subcommand(self):
        bad = self.commands.dispatch(self.event, ["nope"])
        self.assertFalse(bad.ok)
        self.assertIn("Unknown subcommand", bad.text)

    # ---------------------------------------------------------------- 跨群
    def _register_b(self, enabled: bool = True) -> None:
        self.commands.dispatch(
            self.event, ["register", "receive", GROUP_B_ID], is_admin=True
        )
        if not enabled:
            self.commands.dispatch(self.event, ["off", GROUP_B_ID], is_admin=True)

    def test_admin_registers_another_group_from_a(self):
        self._register_b()
        group_b = self.storage.get_group(GROUP_B)
        self.assertIsNotNone(group_b)
        self.assertIs(group_b.direction, Direction.RECEIVE)

    def test_admin_pauses_and_resumes_another_group(self):
        self._register_b()
        self.assertTrue(self.storage.get_group(GROUP_B).enabled)

        off = self.commands.dispatch(self.event, ["off", GROUP_B_ID], is_admin=True)
        self.assertTrue(off.ok, off.text)
        self.assertIn("paused", off.text)
        self.assertFalse(self.storage.get_group(GROUP_B).enabled)

        on = self.commands.dispatch(self.event, ["on", GROUP_B_ID], is_admin=True)
        self.assertTrue(on.ok, on.text)
        self.assertIn("enabled", on.text)
        self.assertTrue(self.storage.get_group(GROUP_B).enabled)

    def test_a_group_managing_b_requires_admin(self):
        self._register_b()
        denied = self.commands.dispatch(self.event, ["off", GROUP_B_ID], is_admin=False)
        self.assertFalse(denied.ok)
        self.assertIn("admin", denied.text.lower())
        # B 群不应被改动
        self.assertTrue(self.storage.get_group(GROUP_B).enabled)

    def test_non_admin_can_still_manage_own_group(self):
        self.commands.dispatch(self.event, ["register", "both"])
        result = self.commands.dispatch(self.event, ["off"], is_admin=False)
        self.assertTrue(result.ok, result.text)
        self.assertFalse(self.storage.get_group(GROUP_A).enabled)

    def test_target_by_full_umo_and_index(self):
        self._register_b()
        by_umo = self.commands.dispatch(
            self.event, ["status", GROUP_B], is_admin=True
        )
        self.assertIn(GROUP_B, by_umo.text)

        by_index = self.commands.dispatch(self.event, ["status", "#1"], is_admin=True)
        self.assertIn("group", by_index.text)

    def test_threshold_with_target(self):
        self._register_b()
        result = self.commands.dispatch(
            self.event, ["threshold", "0.9", GROUP_B_ID], is_admin=True
        )
        self.assertTrue(result.ok, result.text)
        self.assertAlmostEqual(self.storage.get_group(GROUP_B).threshold, 0.9, places=3)

    def test_purge_is_admin_only(self):
        self._register_b()
        denied = self.commands.dispatch(self.event, ["purge", GROUP_B_ID], is_admin=False)
        self.assertFalse(denied.ok)
        purge = self.commands.dispatch(self.event, ["purge", GROUP_B_ID], is_admin=True)
        self.assertTrue(purge.ok, purge.text)
        self.assertIsNone(self.storage.get_group(GROUP_B))

    def test_unknown_target_reports_clearly(self):
        result = self.commands.dispatch(self.event, ["off", "999999"], is_admin=True)
        self.assertFalse(result.ok)
        self.assertIn("not found", result.text.lower())


if __name__ == "__main__":
    unittest.main()
