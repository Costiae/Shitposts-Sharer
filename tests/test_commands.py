"""Command layer tests: all subcommands are ASCII and ASCII-only input works."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.commands import HELP_TEXT, Commands  # noqa: E402
from core.config import DEFAULTS, Settings  # noqa: E402
from core.models import Direction  # noqa: E402
from core.storage import Storage  # noqa: E402
from tests.test_core import scratch_path  # noqa: E402


class FakeEvent:
    def __init__(self, origin: str = "aiocqhttp:GroupMessage:123", name: str = "test group"):
        self.unified_msg_origin = origin
        self.message_obj = type("Obj", (), {"group_name": name})()

    def get_group_id(self):
        return "123"


class TestCommands(unittest.TestCase):
    def setUp(self):
        self.db = scratch_path("commands")
        self.storage = Storage(self.db)
        self.settings = Settings(dict(DEFAULTS))
        self.commands = Commands(self.storage, self.settings)
        self.event = FakeEvent()

    def tearDown(self):
        self.storage.close()
        for suffix in ("", "-wal", "-shm"):
            Path(str(self.db) + suffix).unlink(missing_ok=True)

    def test_help_is_ascii(self):
        HELP_TEXT.encode("ascii")  # raises if any non-ASCII slipped in

    def test_subcommands_are_ascii(self):
        for action in (
            "register", "unregister", "status", "list", "threshold", "pause", "resume",
            "help", "reg", "add", "del", "rm", "ls", "on", "off", "start", "stop", "h",
        ):
            action.encode("ascii")

    def test_register_status_pause_resume(self):
        result = self.commands.dispatch(self.event, ["register", "both"])
        self.assertTrue(result.ok)
        self.assertIn("registered", result.text)

        group = self.storage.get_group(self.event.unified_msg_origin)
        self.assertIs(group.direction, Direction.BOTH)

        status = self.commands.dispatch(self.event, ["status"])
        self.assertIn("confidence", status.text)

        paused = self.commands.dispatch(self.event, ["pause"])
        self.assertIn("paused", paused.text)
        self.assertFalse(self.storage.get_group(self.event.unified_msg_origin).enabled)

        resumed = self.commands.dispatch(self.event, ["resume"])
        self.assertIn("resumed", resumed.text)

    def test_register_direction_aliases(self):
        for token, expected in (("in", Direction.IN), ("out", Direction.OUT), ("both", Direction.BOTH)):
            self.commands.dispatch(self.event, ["register", token])
            group = self.storage.get_group(self.event.unified_msg_origin)
            self.assertIs(group.direction, expected, token)

    def test_threshold_percent(self):
        self.commands.dispatch(self.event, ["register", "in", "80%"])
        group = self.storage.get_group(self.event.unified_msg_origin)
        self.assertAlmostEqual(group.threshold, 0.8, places=3)

    def test_unregister(self):
        self.commands.dispatch(self.event, ["register"])
        result = self.commands.dispatch(self.event, ["unregister"])
        self.assertTrue(result.ok)
        self.assertIsNone(self.storage.get_group(self.event.unified_msg_origin))

    def test_list_and_unknown(self):
        self.assertEqual(self.commands.dispatch(self.event, ["list"]).text, "No group registered yet.")
        bad = self.commands.dispatch(self.event, ["nope"])
        self.assertFalse(bad.ok)
        self.assertIn("Unknown subcommand", bad.text)

    def test_bad_threshold_rejected(self):
        self.commands.dispatch(self.event, ["register"])
        result = self.commands.dispatch(self.event, ["threshold", "abc"])
        self.assertFalse(result.ok)


if __name__ == "__main__":
    unittest.main()
