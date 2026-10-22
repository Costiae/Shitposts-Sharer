"""Shared test doubles for the command layer."""

from __future__ import annotations


class FakeEvent:
    """Minimal AstrMessageEvent stand-in."""

    def __init__(self, origin: str = "aiocqhttp:GroupMessage:100001", name: str = "group A"):
        self.unified_msg_origin = origin
        self.message_str = ""
        self.message_obj = type("Obj", (), {"group_name": name})()
        self.role = "member"

    def get_group_id(self):
        return self.unified_msg_origin.rsplit(":", 1)[-1]

    def plain_result(self, text: str):
        return {"plain": text}
