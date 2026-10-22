"""Storage migration tests: legacy direction values must be upgraded.

Old installs stored direction as 'in'/'out'. The values are now
'receive'/'emit' so the field matches can_receive/can_emit and nobody has
to guess who the "in" refers to. Existing groups must keep working
without re-registering.
"""

from __future__ import annotations

import sqlite3
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.models import Direction  # noqa: E402
from core.storage import Storage  # noqa: E402
from tests.test_core import scratch_path  # noqa: E402


class TestDirectionMigration(unittest.TestCase):
    def setUp(self):
        self.db = scratch_path("migrate")
        for suffix in ("", "-wal", "-shm"):
            Path(str(self.db) + suffix).unlink(missing_ok=True)

    def tearDown(self):
        for suffix in ("", "-wal", "-shm"):
            Path(str(self.db) + suffix).unlink(missing_ok=True)

    def _seed_legacy(self, rows):
        """直接写旧值，绕过 Storage 的新值写入。"""
        storage = Storage(self.db)
        for key, direction in rows:
            storage._conn.execute(
                "INSERT INTO groups(group_key, name, direction, created_at) "
                "VALUES(?, ?, ?, 0)",
                (key, key.upper(), direction),
            )
        storage._conn.commit()
        storage.close()

    def test_legacy_in_becomes_receive(self):
        self._seed_legacy([("g-in", "in")])
        storage = Storage(self.db)
        try:
            self.assertIs(storage.get_group("g-in").direction, Direction.RECEIVE)
        finally:
            storage.close()

    def test_legacy_out_becomes_emit(self):
        self._seed_legacy([("g-out", "out")])
        storage = Storage(self.db)
        try:
            self.assertIs(storage.get_group("g-out").direction, Direction.EMIT)
        finally:
            storage.close()

    def test_both_is_untouched(self):
        self._seed_legacy([("g-both", "both")])
        storage = Storage(self.db)
        try:
            self.assertIs(storage.get_group("g-both").direction, Direction.BOTH)
        finally:
            storage.close()

    def test_migration_is_idempotent(self):
        self._seed_legacy([("g-in", "in"), ("g-out", "out")])
        for _ in range(3):
            storage = Storage(self.db)
            storage.close()
        storage = Storage(self.db)
        try:
            self.assertIs(storage.get_group("g-in").direction, Direction.RECEIVE)
            self.assertIs(storage.get_group("g-out").direction, Direction.EMIT)
        finally:
            storage.close()

    def test_raw_values_are_rewritten(self):
        self._seed_legacy([("g-in", "in"), ("g-out", "out")])
        storage = Storage(self.db)
        try:
            rows = storage._conn.execute(
                "SELECT group_key, direction FROM groups ORDER BY group_key"
            ).fetchall()
            self.assertEqual(
                [(r["group_key"], r["direction"]) for r in rows],
                [("g-in", "receive"), ("g-out", "emit")],
            )
        finally:
            storage.close()

    def test_list_groups_by_direction_after_migration(self):
        self._seed_legacy([("g-in", "in"), ("g-out", "out"), ("g-both", "both")])
        storage = Storage(self.db)
        try:
            receivers = {
                g.group_key for g in storage.list_groups(Direction.RECEIVE)
            }
            emitters = {g.group_key for g in storage.list_groups(Direction.EMIT)}
            self.assertEqual(receivers, {"g-in", "g-both"})
            self.assertEqual(emitters, {"g-out", "g-both"})
        finally:
            storage.close()

    def test_mixed_legacy_and_new_rows_coexist(self):
        """迁移后新写入的新值不会被误改。"""
        self._seed_legacy([("g-in", "in")])
        storage = Storage(self.db)
        try:
            from core.models import GroupProfile

            storage.upsert_group(
                GroupProfile(group_key="g-new", name="new", direction=Direction.EMIT)
            )
        finally:
            storage.close()

        storage = Storage(self.db)
        try:
            self.assertIs(storage.get_group("g-in").direction, Direction.RECEIVE)
            self.assertIs(storage.get_group("g-new").direction, Direction.EMIT)
        finally:
            storage.close()


class TestDriftColumnMigration(unittest.TestCase):
    """老库的 groups 表没有 drift 列，加载时必须自动 ALTER 补上。"""

    def setUp(self):
        self.db = scratch_path("migrate-drift")
        for suffix in ("", "-wal", "-shm"):
            Path(str(self.db) + suffix).unlink(missing_ok=True)

    def tearDown(self):
        for suffix in ("", "-wal", "-shm"):
            Path(str(self.db) + suffix).unlink(missing_ok=True)

    def _seed_v1_schema(self):
        conn = sqlite3.connect(self.db)
        conn.executescript(
            """
            CREATE TABLE groups (
                group_key            TEXT PRIMARY KEY,
                name                 TEXT NOT NULL DEFAULT '',
                direction            TEXT NOT NULL DEFAULT 'both',
                intake_json          TEXT NOT NULL DEFAULT '{}',
                threshold            REAL NOT NULL DEFAULT 0,
                enabled              INTEGER NOT NULL DEFAULT 1,
                native_message_count INTEGER NOT NULL DEFAULT 0,
                native_char_count    INTEGER NOT NULL DEFAULT 0,
                accepted_count       INTEGER NOT NULL DEFAULT 0,
                rejected_count       INTEGER NOT NULL DEFAULT 0,
                last_shit_at         REAL NOT NULL DEFAULT 0,
                created_at           REAL NOT NULL DEFAULT 0,
                updated_at           REAL NOT NULL DEFAULT 0
            );
            INSERT INTO groups(group_key, name, direction)
            VALUES('legacy', 'legacy', 'receive');
            """
        )
        conn.commit()
        conn.close()

    def test_legacy_table_gets_drift_column(self):
        self._seed_v1_schema()
        storage = Storage(self.db)
        try:
            group = storage.get_group("legacy")
            self.assertIsNotNone(group)
            self.assertEqual(group.drift, 0.0)
            self.assertEqual(storage.read_setting("schema_version"), "2")
        finally:
            storage.close()

    def test_migration_is_idempotent(self):
        self._seed_v1_schema()
        for _ in range(3):
            Storage(self.db).close()
        storage = Storage(self.db)
        try:
            self.assertEqual(storage.get_group("legacy").drift, 0.0)
        finally:
            storage.close()


if __name__ == "__main__":
    unittest.main()
