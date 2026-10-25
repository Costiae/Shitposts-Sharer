"""SQLite 持久化层。

设计要点：
- 单库多表（群、屎、搬运记录、设置），库文件放在 AstrBot 的 data 目录下。
- 两个独立连接：``connect()`` 给业务（异步线程里跑同步 IO），``connect_ro()``
  给 WebUI 的只读 API，避免面板查询把写入卡住。
- 所有写操作通过 ``_write()`` 串行化，SQLite 本身是单写者。
"""

from __future__ import annotations

import json
import sqlite3
import threading
import time
from pathlib import Path
from typing import Any, Iterable

from .models import Direction, GroupProfile, ShitRecord
from .vector import normalize_intake_terms

SCHEMA_VERSION = 6

_SCHEMA = """
CREATE TABLE IF NOT EXISTS groups (
    group_key            TEXT PRIMARY KEY,
    name                 TEXT NOT NULL DEFAULT '',
    direction            TEXT NOT NULL DEFAULT 'both',
    intake_json          TEXT NOT NULL DEFAULT '{}',
    threshold            REAL NOT NULL DEFAULT 0,
    enabled              INTEGER NOT NULL DEFAULT 1,
    image_enabled        INTEGER NOT NULL DEFAULT 1,
    drift                REAL NOT NULL DEFAULT 0,
    native_message_count INTEGER NOT NULL DEFAULT 0,
    native_char_count    INTEGER NOT NULL DEFAULT 0,
    accepted_count       INTEGER NOT NULL DEFAULT 0,
    rejected_count       INTEGER NOT NULL DEFAULT 0,
    last_shit_at         REAL NOT NULL DEFAULT 0,
    created_at           REAL NOT NULL DEFAULT 0,
    updated_at           REAL NOT NULL DEFAULT 0
);

CREATE TABLE IF NOT EXISTS shits (
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
    image_tags_json   TEXT NOT NULL DEFAULT '[]',
    image_vector_json TEXT NOT NULL DEFAULT '{}',
    nsfw          INTEGER NOT NULL DEFAULT 0,
    intensity     REAL NOT NULL DEFAULT 0,
    context_included INTEGER NOT NULL DEFAULT 0,
    payload_json  TEXT NOT NULL DEFAULT '[]',
    context_text  TEXT NOT NULL DEFAULT '',
    reason        TEXT NOT NULL DEFAULT ''
);

CREATE INDEX IF NOT EXISTS idx_shits_fp  ON shits(fingerprint);
CREATE INDEX IF NOT EXISTS idx_shits_src ON shits(source_group, created_at);

CREATE TABLE IF NOT EXISTS deliveries (
    id         INTEGER PRIMARY KEY AUTOINCREMENT,
    shit_id    TEXT NOT NULL,
    group_key  TEXT NOT NULL,
    score      REAL NOT NULL DEFAULT 0,
    confidence REAL NOT NULL DEFAULT 0,
    source     TEXT NOT NULL DEFAULT '',
    status     TEXT NOT NULL DEFAULT 'ok',
    created_at REAL NOT NULL
);

CREATE UNIQUE INDEX IF NOT EXISTS idx_deliveries_once
    ON deliveries(shit_id, group_key);

CREATE INDEX IF NOT EXISTS idx_deliveries_group
    ON deliveries(group_key, created_at);

CREATE TABLE IF NOT EXISTS settings (
    key   TEXT PRIMARY KEY,
    value TEXT NOT NULL
);
"""


class Storage:
    def __init__(self, db_path: str | Path) -> None:
        self.path = Path(db_path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        self._conn = sqlite3.connect(self.path, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("PRAGMA synchronous=NORMAL")
        self._write(self._conn.executescript, _SCHEMA)
        self._migrate()

    # ------------------------------------------------------------------ 连接
    def connect_ro(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.path, check_same_thread=False)
        conn.row_factory = sqlite3.Row
        return conn

    def _write(self, func, *args, **kwargs):
        with self._lock:
            result = func(*args, **kwargs)
            self._conn.commit()
            return result

    def close(self) -> None:
        with self._lock:
            self._conn.close()

    def _migrate(self) -> None:
        self._migrate_group_columns()
        self._migrate_shit_columns()
        self._migrate_direction_values()
        try:
            version = int(self.read_setting("schema_version", "0") or 0)
        except ValueError:
            version = 0
        if version < 6:
            self._migrate_intake_vectors()
        if version < SCHEMA_VERSION:
            self.write_setting("schema_version", str(SCHEMA_VERSION))

    def _ensure_columns(self, table: str, columns: dict[str, str]) -> None:
        """按需给已存在的表补列（ALTER TABLE ADD COLUMN）。"""
        existing = {
            row["name"]
            for row in self._conn.execute(f"PRAGMA table_info({table})").fetchall()
        }
        changed = False
        for name, ddl in columns.items():
            if name not in existing:
                self._conn.execute(f"ALTER TABLE {table} ADD COLUMN {name} {ddl}")
                changed = True
        if changed:
            self._conn.commit()

    def _migrate_group_columns(self) -> None:
        """老库补 groups 列：口味偏移度与图片开关。"""
        self._ensure_columns(
            "groups",
            {
                "drift": "REAL NOT NULL DEFAULT 0",
                "image_enabled": "INTEGER NOT NULL DEFAULT 1",
            },
        )

    def _migrate_shit_columns(self) -> None:
        """shits 的图片标签列（老库默认空，图片按原关键词逻辑匹配）。"""
        self._ensure_columns(
            "shits",
            {
                "image_tags_json": "TEXT NOT NULL DEFAULT '[]'",
                "image_vector_json": "TEXT NOT NULL DEFAULT '{}'",
                "nsfw": "INTEGER NOT NULL DEFAULT 0",
            },
        )

    def _migrate_direction_values(self) -> None:
        """in/out -> receive/emit。

        旧版本把群的角色叫 in / out，主语有歧义（到底是谁 in？）。
        现在字段值改成 receive / emit，与 can_receive / can_emit 对齐。
        已注册的群必须无感升级，否则它们的角色会读不出来。
        """
        mapping = {"in": "receive", "out": "emit"}
        for old, new in mapping.items():
            updated = self._conn.execute(
                "UPDATE groups SET direction = ? WHERE direction = ?", (new, old)
            ).rowcount
            if updated:
                self._write(lambda: None)  # 触发 commit
        self._conn.commit()

    def _migrate_intake_vectors(self) -> None:
        """把旧版/外部写入的口味表压成当前整词空间。

        旧数据可能混有 2 字碎片（``二次``/``次元``/``站``）、停用词
        （``玩梗``）、未归一的别名（``AI``）或邀请码/哈希。当前向量按
        整词计权，这些残留会稀释或污染匹配；启动时统一清理一次，避免
        老群明明有口味却匹配不到。
        """
        rows = self._conn.execute(
            "SELECT group_key, intake_json FROM groups"
        ).fetchall()
        changed = False
        for row in rows:
            try:
                raw = json.loads(row["intake_json"] or "{}")
            except (TypeError, ValueError):
                continue
            if not isinstance(raw, dict) or not raw:
                continue
            normalized = normalize_intake_terms(raw)
            if normalized != raw:
                self._conn.execute(
                    "UPDATE groups SET intake_json = ? WHERE group_key = ?",
                    (
                        json.dumps(normalized, ensure_ascii=False),
                        row["group_key"],
                    ),
                )
                changed = True
        if changed:
            self._conn.commit()

    # ------------------------------------------------------------------ 设置
    def read_setting(self, key: str, default: str = "") -> str:
        row = self._conn.execute(
            "SELECT value FROM settings WHERE key = ?", (key,)
        ).fetchone()
        return row["value"] if row else default

    def write_setting(self, key: str, value: str) -> None:
        self._write(
            self._conn.execute,
            "INSERT INTO settings(key, value) VALUES(?, ?) "
            "ON CONFLICT(key) DO UPDATE SET value = excluded.value",
            (key, value),
        )

    # ------------------------------------------------------------------ 群
    @staticmethod
    def _row_to_group(row: sqlite3.Row) -> GroupProfile:
        return GroupProfile(
            group_key=row["group_key"],
            name=row["name"],
            direction=Direction(row["direction"]),
            intake=json.loads(row["intake_json"] or "{}"),
            threshold=float(row["threshold"]),
            enabled=bool(row["enabled"]),
            image_enabled=bool(row["image_enabled"]),
            drift=float(row["drift"]),
            native_message_count=int(row["native_message_count"]),
            native_char_count=int(row["native_char_count"]),
            accepted_count=int(row["accepted_count"]),
            rejected_count=int(row["rejected_count"]),
            last_shit_at=float(row["last_shit_at"]),
            created_at=float(row["created_at"]),
            updated_at=float(row["updated_at"]),
        )

    def get_group(self, group_key: str) -> GroupProfile | None:
        row = self._conn.execute(
            "SELECT * FROM groups WHERE group_key = ?", (group_key,)
        ).fetchone()
        return self._row_to_group(row) if row else None

    def list_groups(self, direction: Direction | None = None) -> list[GroupProfile]:
        sql = "SELECT * FROM groups"
        params: tuple[Any, ...] = ()
        if direction is not None:
            # both 同时满足 receive 与 emit 的查询
            if direction in (Direction.RECEIVE, Direction.EMIT):
                sql += " WHERE direction IN (?, 'both')"
                params = (direction.value,)
            else:
                sql += " WHERE direction = ?"
                params = (direction.value,)
        sql += " ORDER BY created_at ASC"
        rows = self._conn.execute(sql, params).fetchall()
        return [self._row_to_group(row) for row in rows]

    def upsert_group(self, profile: GroupProfile) -> GroupProfile:
        profile.updated_at = time.time()
        self._write(
            self._conn.execute,
            """
            INSERT INTO groups (
                group_key, name, direction, intake_json, threshold, enabled,
                image_enabled, drift, native_message_count, native_char_count,
                accepted_count, rejected_count, last_shit_at, created_at,
                updated_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(group_key) DO UPDATE SET
                name = excluded.name,
                direction = excluded.direction,
                intake_json = excluded.intake_json,
                threshold = excluded.threshold,
                enabled = excluded.enabled,
                image_enabled = excluded.image_enabled,
                drift = excluded.drift,
                native_message_count = excluded.native_message_count,
                native_char_count = excluded.native_char_count,
                accepted_count = excluded.accepted_count,
                rejected_count = excluded.rejected_count,
                last_shit_at = excluded.last_shit_at,
                updated_at = excluded.updated_at
            """,
            (
                profile.group_key,
                profile.name,
                profile.direction.value,
                json.dumps(profile.intake, ensure_ascii=False),
                profile.threshold,
                int(profile.enabled),
                int(profile.image_enabled),
                profile.drift,
                profile.native_message_count,
                profile.native_char_count,
                profile.accepted_count,
                profile.rejected_count,
                profile.last_shit_at,
                profile.created_at,
                profile.updated_at,
            ),
        )
        return profile

    def delete_group(self, group_key: str) -> bool:
        cursor = self._write(
            self._conn.execute, "DELETE FROM groups WHERE group_key = ?", (group_key,)
        )
        return cursor.rowcount > 0

    def registered(self, group_key: str) -> bool:
        return self.get_group(group_key) is not None

    # ------------------------------------------------------------------ 屎
    @staticmethod
    def _row_to_shit(row: sqlite3.Row) -> ShitRecord:
        return ShitRecord(
            shit_id=row["shit_id"],
            source_group=row["source_group"],
            ts=float(row["created_at"]),
            kind=Kind_or_text(row["kind"]),
            text_blob=row["text_blob"],
            keywords=json.loads(row["keywords_json"] or "[]"),
            vector=json.loads(row["vector_json"] or "{}"),
            fingerprint=row["fingerprint"],
            incoming=bool(row["incoming"]),
            tags=json.loads(row["tags_json"] or "[]"),
            image_tags=json.loads(row["image_tags_json"] or "[]"),
            image_vector=json.loads(row["image_vector_json"] or "{}"),
            nsfw=bool(row["nsfw"]),
            intensity=float(row["intensity"]),
            context_included=bool(row["context_included"]),
            payload=json.loads(row["payload_json"] or "[]"),
            context_text=row["context_text"],
            reason=row["reason"],
        )

    def add_shit(self, record: ShitRecord) -> None:
        self._write(
            self._conn.execute,
            """
            INSERT OR REPLACE INTO shits (
                shit_id, source_group, created_at, kind, text_blob, keywords_json,
                vector_json, fingerprint, incoming, tags_json, image_tags_json,
                image_vector_json, nsfw, intensity, context_included,
                payload_json, context_text, reason
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                record.shit_id,
                record.source_group,
                record.ts,
                record.kind.value,
                record.text_blob,
                json.dumps(record.keywords, ensure_ascii=False),
                json.dumps(record.vector, ensure_ascii=False),
                record.fingerprint,
                int(record.incoming),
                json.dumps(record.tags, ensure_ascii=False),
                json.dumps(record.image_tags, ensure_ascii=False),
                json.dumps(record.image_vector, ensure_ascii=False),
                int(record.nsfw),
                record.intensity,
                int(record.context_included),
                json.dumps(record.payload, ensure_ascii=False),
                record.context_text,
                record.reason,
            ),
        )

    def get_shit(self, shit_id: str) -> ShitRecord | None:
        row = self._conn.execute(
            "SELECT * FROM shits WHERE shit_id = ?", (shit_id,)
        ).fetchone()
        return self._row_to_shit(row) if row else None

    def find_by_fingerprint(self, fingerprint: str) -> ShitRecord | None:
        row = self._conn.execute(
            "SELECT * FROM shits WHERE fingerprint = ? ORDER BY created_at DESC LIMIT 1",
            (fingerprint,),
        ).fetchone()
        return self._row_to_shit(row) if row else None

    def iter_vectors(self, since: float = 0.0) -> Iterable[tuple[str, dict[str, float]]]:
        rows = self._conn.execute(
            "SELECT shit_id, vector_json FROM shits WHERE created_at >= ? "
            "ORDER BY created_at DESC LIMIT 2000",
            (since,),
        ).fetchall()
        for row in rows:
            yield row["shit_id"], json.loads(row["vector_json"] or "{}")

    # ------------------------------------------------------------------ 搬运
    def record_delivery(
        self,
        shit_id: str,
        group_key: str,
        score: float,
        confidence: float,
        source: str,
        status: str = "ok",
    ) -> bool:
        """记录一次搬运；同一条屎对同一个群只会成功记录一次。"""
        try:
            self._write(
                self._conn.execute,
                "INSERT INTO deliveries "
                "(shit_id, group_key, score, confidence, source, status, created_at) "
                "VALUES (?, ?, ?, ?, ?, ?, ?)",
                (shit_id, group_key, score, confidence, source, status, time.time()),
            )
        except sqlite3.IntegrityError:
            return False
        return True

    def already_delivered(self, shit_id: str, group_key: str) -> bool:
        row = self._conn.execute(
            "SELECT 1 FROM deliveries WHERE shit_id = ? AND group_key = ? LIMIT 1",
            (shit_id, group_key),
        ).fetchone()
        return row is not None

    def delivered_groups(self, shit_id: str) -> list[tuple[str, str]]:
        rows = self._conn.execute(
            "SELECT group_key, status FROM deliveries WHERE shit_id = ? "
            "ORDER BY created_at ASC",
            (shit_id,),
        ).fetchall()
        return [(row["group_key"], row["status"]) for row in rows]

    def mark_delivery(self, shit_id: str, group_key: str, status: str) -> None:
        self._write(
            self._conn.execute,
            "UPDATE deliveries SET status = ? WHERE shit_id = ? AND group_key = ?",
            (status, shit_id, group_key),
        )

    def recent_deliveries(self, group_key: str, limit: int = 20) -> list[dict[str, Any]]:
        rows = self._conn.execute(
            "SELECT shit_id, score, confidence, source, status, created_at "
            "FROM deliveries WHERE group_key = ? ORDER BY created_at DESC LIMIT ?",
            (group_key, limit),
        ).fetchall()
        return [dict(row) for row in rows]

    def stats(self) -> dict[str, int]:
        groups = self._conn.execute("SELECT COUNT(*) AS c FROM groups").fetchone()["c"]
        shits = self._conn.execute("SELECT COUNT(*) AS c FROM shits").fetchone()["c"]
        deliveries = self._conn.execute(
            "SELECT COUNT(*) AS c FROM deliveries"
        ).fetchone()["c"]
        return {"groups": groups, "shits": shits, "deliveries": deliveries}


def Kind_or_text(raw: str):
    """兼容历史数据：未知类型一律当作文本。"""
    from .models import Kind

    try:
        return Kind(raw)
    except ValueError:
        return Kind.TEXT
