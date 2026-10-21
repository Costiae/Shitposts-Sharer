"""群指令：注册群、看口味、调阈值、暂停。

设计原则：**群友随时能自己注册自己的群**，因为"群里一套、WebUI 一套"
会很烦；但**修改别人的群、全局开关、手动强制搬**只给管理员。

所有指令与子指令一律使用 ASCII 英文，避免不同平台/编码下的解析歧义。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .config import Settings
from .models import Direction, GroupProfile
from .profile import confidence_score, describe_intake
from .storage import Storage

HELP_TEXT = """Shitposts Sharer commands
/shit register <in|out|both> [threshold]   join the network (default: both)
/shit unregister                           leave the network
/shit status                               this group's taste and confidence
/shit list                                 all registered groups
/shit threshold <0-1|xx%>                  delivery threshold for this group
/shit pause | /shit resume                 pause / resume this group
/shit help                                 show this help

Direction: in = receive only, out = emit only, both = receive and emit."""


@dataclass(slots=True)
class CommandResult:
    text: str
    ok: bool = True


class Commands:
    def __init__(self, storage: Storage, settings: Settings) -> None:
        self.storage = storage
        self.settings = settings

    # ------------------------------------------------------------------ 主分发
    def dispatch(
        self, event: Any, args: list[str], *, is_admin: bool = False
    ) -> CommandResult:
        if not args:
            return CommandResult(HELP_TEXT)

        action = args[0].strip().lower()
        rest = args[1:]

        if action in ("register", "reg", "add"):
            return self.register(event, rest, is_admin=is_admin)
        if action in ("unregister", "del", "remove", "rm"):
            return self.unregister(event, is_admin=is_admin)
        if action in ("status", "info"):
            return self.status(event)
        if action in ("list", "ls", "groups"):
            return self.list_groups()
        if action in ("threshold", "thr"):
            return self.set_threshold(event, rest, is_admin=is_admin)
        if action in ("pause", "off", "stop"):
            return self.toggle(event, enabled=False, is_admin=is_admin)
        if action in ("resume", "on", "start"):
            return self.toggle(event, enabled=True, is_admin=is_admin)
        if action in ("help", "h", "?"):
            return CommandResult(HELP_TEXT)
        return CommandResult(f"Unknown subcommand: {action}\n\n{HELP_TEXT}", ok=False)

    # ------------------------------------------------------------------ 子指令
    def register(
        self, event: Any, args: list[str], *, is_admin: bool = False
    ) -> CommandResult:
        group_key = unified_origin(event)
        if not group_key:
            return CommandResult("Only usable inside a group chat.", ok=False)

        raw_direction = args[0] if args else Direction.BOTH.value
        try:
            direction = Direction.parse(raw_direction)
        except ValueError as exc:
            return CommandResult(f"{exc} - use in / out / both", ok=False)

        threshold = 0.0
        if len(args) > 1:
            try:
                threshold = self.settings.parse_threshold(args[1])
            except ValueError:
                return CommandResult(
                    "Threshold must be 0-1 or a percentage, e.g. 0.8 or 80%", ok=False
                )

        group = self.storage.get_group(group_key)
        name = group_name(event)
        if group is None:
            group = GroupProfile(
                group_key=group_key,
                name=name,
                direction=direction,
                threshold=threshold,
            )
        else:
            group.direction = direction
            group.name = name or group.name
            if len(args) > 1:
                group.threshold = threshold
        self.storage.upsert_group(group)

        confidence = confidence_score(
            native_messages=group.native_message_count,
            native_chars=group.native_char_count,
            drift=0.0,
            accepted=group.accepted_count,
            rejected=group.rejected_count,
            k=self.settings.confidence_k,
        )
        effective = self.settings.threshold_for(group.threshold)
        return CommandResult(
            f"[ok] registered: {group.name or group_key}\n"
            f"direction : {group.direction.value}\n"
            f"confidence: {confidence:.2f} (threshold {effective:.2f})\n"
            f"taste     : {describe_intake(group)}\n"
            f"\n"
            f"Confidence accrues from this group's own (native) shit samples.\n"
            f"Only once it clears the threshold will it start receiving."
        )

    def unregister(self, event: Any, *, is_admin: bool = False) -> CommandResult:
        group_key = unified_origin(event)
        if not group_key:
            return CommandResult("Only usable inside a group chat.", ok=False)
        if self.storage.delete_group(group_key):
            return CommandResult("[ok] this group left the network.")
        return CommandResult("This group was not registered.", ok=False)

    def status(self, event: Any) -> CommandResult:
        group_key = unified_origin(event)
        group = self.storage.get_group(group_key) if group_key else None
        if group is None:
            return CommandResult(
                "This group is not registered. Run /shit register both to join.",
                ok=False,
            )

        confidence = confidence_score(
            native_messages=group.native_message_count,
            native_chars=group.native_char_count,
            drift=0.0,
            accepted=group.accepted_count,
            rejected=group.rejected_count,
            k=self.settings.confidence_k,
        )
        threshold = self.settings.threshold_for(group.threshold)
        state = "running" if group.enabled else "paused"
        verdict = "eligible" if confidence >= threshold else "not enough samples"
        return CommandResult(
            f"group     : {group.name or group_key}\n"
            f"direction : {group.direction.value} ({state})\n"
            f"confidence: {confidence:.2f} (threshold {threshold:.2f}) -> {verdict}\n"
            f"native    : {group.native_message_count} shit(s), {group.native_char_count} chars\n"
            f"accepted  : {group.accepted_count}  rejected: {group.rejected_count}\n"
            f"taste     : {describe_intake(group)}"
        )

    def list_groups(self) -> CommandResult:
        groups = self.storage.list_groups()
        if not groups:
            return CommandResult("No group registered yet.")
        lines = [f"{len(groups)} group(s):"]
        for group in groups:
            confidence = confidence_score(
                native_messages=group.native_message_count,
                native_chars=group.native_char_count,
                drift=0.0,
                accepted=group.accepted_count,
                rejected=group.rejected_count,
                k=self.settings.confidence_k,
            )
            lines.append(
                f"- {group.name or group.group_key} | {group.direction.value} | "
                f"conf {confidence:.2f} | native {group.native_message_count}"
            )
        return CommandResult("\n".join(lines))

    def set_threshold(
        self, event: Any, args: list[str], *, is_admin: bool = False
    ) -> CommandResult:
        group_key = unified_origin(event)
        group = self.storage.get_group(group_key) if group_key else None
        if group is None:
            return CommandResult("This group is not registered.", ok=False)
        if not args:
            current = self.settings.threshold_for(group.threshold)
            return CommandResult(f"Current threshold: {current:.2f}")
        try:
            value = self.settings.parse_threshold(args[0])
        except ValueError:
            return CommandResult(
                "Threshold must be 0-1 or a percentage, e.g. 0.8 or 80%", ok=False
            )
        group.threshold = value
        self.storage.upsert_group(group)
        return CommandResult(
            f"[ok] threshold set to {self.settings.threshold_for(value):.2f}"
        )

    def toggle(
        self, event: Any, *, enabled: bool, is_admin: bool = False
    ) -> CommandResult:
        group_key = unified_origin(event)
        group = self.storage.get_group(group_key) if group_key else None
        if group is None:
            return CommandResult("This group is not registered.", ok=False)
        group.enabled = enabled
        self.storage.upsert_group(group)
        return CommandResult(
            "[ok] resumed this group." if enabled else "[ok] paused this group."
        )


# ---------------------------------------------------------------------- 工具
def unified_origin(event: Any) -> str:
    return str(getattr(event, "unified_msg_origin", "") or "")


def group_name(event: Any) -> str:
    obj = getattr(event, "message_obj", None)
    name = getattr(obj, "group_name", "") or ""
    if not name:
        group_id = getattr(event, "get_group_id", lambda: "")()
        name = f"group {group_id}" if group_id else "group"
    return str(name)[:60]


def is_group_event(event: Any) -> bool:
    getter = getattr(event, "get_group_id", None)
    if not callable(getter):
        return False
    return bool(getter())
