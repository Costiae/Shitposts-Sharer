"""Storage migration tests: legacy direction values must be upgraded.

Old installs stored direction as 'in'/'out'. The values are now
'receive'/'emit' so the field matches can_receive/can_emit and nobody has
to guess who the "in" refers to. Existing groups must keep working
without re-registering.
"""

from __future__ import annotations

import json
import sqlite3
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.models import Direction, Kind, ShitRecord  # noqa: E402
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
            # 老群默认保持图片搬运开启，行为不变
            self.assertTrue(group.image_enabled)
            self.assertEqual(storage.read_setting("schema_version"), "6")
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


class TestShitImageColumnMigration(unittest.TestCase):
    """老库的 shits 表没有图片标签列，加载时必须自动补齐。"""

    def setUp(self):
        self.db = scratch_path("migrate-shit-image")
        for suffix in ("", "-wal", "-shm"):
            Path(str(self.db) + suffix).unlink(missing_ok=True)

    def tearDown(self):
        for suffix in ("", "-wal", "-shm"):
            Path(str(self.db) + suffix).unlink(missing_ok=True)

    def _seed_v2_shits_schema(self):
        conn = sqlite3.connect(self.db)
        conn.executescript(
            """
            CREATE TABLE shits (
                shit_id       TEXT PRIMARY KEY,
                source_group  TEXT NOT NULL,
                created_at    REAL NOT NULL,
                kind          TEXT NOT NULL,
                text_blob     TEXT NOT NULL DEFAULT '',
                keywords_json TEXT NOT NULL DEFAULT '[]',
                vector_json   TEXT NOT NULL DEFAULT '{}',
                fingerprint   TEXT NOT NULL DEFAULT '',
                incoming      INTEGER NOT NULL DEFAULT 0,
                tags_json     TEXT NOT NULL DEFAULT '[]',
                intensity     REAL NOT NULL DEFAULT 0,
                context_included INTEGER NOT NULL DEFAULT 0,
                payload_json  TEXT NOT NULL DEFAULT '[]',
                context_text  TEXT NOT NULL DEFAULT '',
                reason        TEXT NOT NULL DEFAULT ''
            );
            """
        )
        conn.commit()
        conn.close()

    def test_old_shits_table_gets_image_columns(self):
        self._seed_v2_shits_schema()
        storage = Storage(self.db)
        try:
            record = ShitRecord(
                shit_id="s-img",
                source_group="g",
                kind=Kind.IMAGE,
                text_blob="图",
                image_tags=["初音未来", "NSFW"],
                image_vector={"初音未来": 0.8, "nsfw": 0.2},
                nsfw=True,
            )
            storage.add_shit(record)
            loaded = storage.get_shit("s-img")
            self.assertEqual(loaded.image_tags, ["初音未来", "NSFW"])
            self.assertTrue(loaded.nsfw)
            self.assertAlmostEqual(loaded.image_vector["初音未来"], 0.8)
        finally:
            storage.close()


class TestIntakeVectorMigration(unittest.TestCase):
    """旧口味表里的碎片/停用词/邀请码必须在升级时清理一次。"""

    def setUp(self):
        self.db = scratch_path("migrate-intake")
        for suffix in ("", "-wal", "-shm"):
            Path(str(self.db) + suffix).unlink(missing_ok=True)

    def tearDown(self):
        for suffix in ("", "-wal", "-shm"):
            Path(str(self.db) + suffix).unlink(missing_ok=True)

    def test_legacy_terms_are_compacted(self):
        raw = {
            "b站": 0.52,
            "鸡8狗10": 0.32,
            "转发图": 0.25,
            "二次元": 0.25,
            "二次": 0.15,
            "次元": 0.15,
            "站": 0.10,
            "玩梗": 0.20,
            "AI": 0.10,
            "角色图": 0.21,
            (
                "l2fj452ueusetgle7i3le8fq4inoas12dhek58ppm3tdaol4jb5jud2036u8"
                "maiic5maf95oe33bpcuir2mqon9dvgfurgqqeoipqnsqvkhggrqem199lrbndl1j1d"
            ): 0.20,
            "抽象": 0.20,
        }
        storage = Storage(self.db)
        storage._conn.execute(
            "INSERT INTO groups(group_key, name, direction, intake_json) "
            "VALUES(?, ?, 'both', ?)",
            ("legacy-intake", "legacy", json.dumps(raw, ensure_ascii=False)),
        )
        storage.write_setting("schema_version", "4")
        storage.close()

        storage = Storage(self.db)
        try:
            group = storage.get_group("legacy-intake")
            self.assertIsNotNone(group)
            assert group is not None
            for term in (
                "b站",
                "转发图",
                "二次元",
                "角色图",
                "人工智能",
                "抽象",
            ):
                self.assertIn(term, group.intake)
            for noise in (
                "二次",
                "次元",
                "站",
                "玩梗",
                "AI",
                "鸡8狗10",
                "l2fj452ueusetgle7i3le8fq4inoas12dhek58ppm3tdaol4jb5jud2036u8"
                "maiic5maf95oe33bpcuir2mqon9dvgfurgqqeoipqnsqvkhggrqem199lrbndl1j1d",
            ):
                self.assertNotIn(noise, group.intake)
            self.assertFalse(any(len(term) > 32 for term in group.intake))
            self.assertEqual(storage.read_setting("schema_version"), "6")
        finally:
            storage.close()

        # 重复启动不应再次放大权重。
        storage = Storage(self.db)
        try:
            again = storage.get_group("legacy-intake")
            assert again is not None
            self.assertEqual(again.intake, group.intake)
        finally:
            storage.close()


if __name__ == "__main__":
    unittest.main()
