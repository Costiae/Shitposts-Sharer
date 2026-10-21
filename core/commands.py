"""群指令：注册群、看口味、调阈值、暂停。

设计原则：**群友随时能自己注册自己的群**，因为"群里一套、WebUI 一套"
会很烦；但**修改别人的群、全局开关、手动强制搬**只给管理员。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .config import Settings
from .models import Direction, GroupProfile
from .profile import confidence_score, describe_intake
from .storage import Storage

HELP_TEXT = """🧻 搬屎插件指令
/屎 注册 <in|out|both> [阈值]   把本群注册进来（收/发/进出）
/屎 注销                        把本群移出搬运网络
/屎 状态                        看本群的口味与可信度
/屎 群列表                      看所有已注册的群
/屎 阈值 <0-1>                  设置本群的搬运阈值
/屎 暂停 / /屎 继续             暂停 / 恢复本群收发
/屎 搬 <文本|回复一条消息>      手动试搬（仅管理员）

方向说明：in=只收别人的屎；out=只往外发；both=既收又发"""


@dataclass(slots=True)
class CommandResult:
    text: str
    ok: bool = True


class Commands:
    def __init__(self, storage: Storage, settings: Settings) -> None:
        self.storage = storage
        self.settings = settings

    # ------------------------------------------------------------------ 主分发
    def dispatch(self, event: Any, args: list[str], *, is_admin: bool = False) -> CommandResult:
        if not args:
            return CommandResult(HELP_TEXT)

        action = args[0].strip().lower()
        rest = args[1:]

        if action in ("注册", "register", "reg"):
            return self.register(event, rest, is_admin=is_admin)
        if action in ("注销", "unregister", "del"):
            return self.unregister(event, is_admin=is_admin)
        if action in ("状态", "status", "info"):
            return self.status(event)
        if action in ("群列表", "列表", "list"):
            return self.list_groups()
        if action in ("阈值", "threshold", "thr"):
            return self.set_threshold(event, rest, is_admin=is_admin)
        if action in ("暂停", "pause"):
            return self.toggle(event, enabled=False, is_admin=is_admin)
        if action in ("继续", "恢复", "resume"):
            return self.toggle(event, enabled=True, is_admin=is_admin)
        if action in ("搬", "test"):
            return CommandResult("手动试搬请在群里回复一条消息发送 /屎 搬", ok=False)
        if action in ("帮助", "help"):
            return CommandResult(HELP_TEXT)
        return CommandResult(f"未知指令：{action}\n\n{HELP_TEXT}", ok=False)

    # ------------------------------------------------------------------ 子指令
    def register(
        self, event: Any, args: list[str], *, is_admin: bool = False
    ) -> CommandResult:
        group_key = unified_origin(event)
        if not group_key:
            return CommandResult("只能在群里注册。", ok=False)

        raw_direction = args[0] if args else (Direction.BOTH.value)
        try:
            direction = Direction.parse(raw_direction)
        except ValueError as exc:
            return CommandResult(f"{exc}，可用 in / out / both", ok=False)

        threshold = 0.0
        if len(args) > 1:
            try:
                threshold = self.settings.parse_threshold(args[1])
            except ValueError:
                return CommandResult("阈值需要是 0-1 之间的小数或百分比，例如 0.8 / 80%", ok=False)

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
            # 改方向是“改自己群”的操作，允许；但降级成 out 会停掉别人的屎
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
            f"✅ 已注册「{group.name or group_key}」\n"
            f"方向：{group.direction.label()}\n"
            f"当前可信度：{confidence:.2f}（阈值 {effective:.2f}）\n"
            f"可信度是从本群**原生屎**的样本量与偏移度累计出来的，"
            f"样本够了才会真正开始收到别人的屎。\n"
            f"口味：{describe_intake(group)}"
        )

    def unregister(self, event: Any, *, is_admin: bool = False) -> CommandResult:
        group_key = unified_origin(event)
        if not group_key:
            return CommandResult("只能在群里注销。", ok=False)
        removed = self.storage.delete_group(group_key)
        if removed:
            return CommandResult("🗑️ 已把本群移出搬运网络。")
        return CommandResult("本群本来就没注册。", ok=False)

    def status(self, event: Any) -> CommandResult:
        group_key = unified_origin(event)
        group = self.storage.get_group(group_key) if group_key else None
        if group is None:
            return CommandResult(
                "本群还没注册。发送 /屎 注册 both 即可加入搬运网络。", ok=False
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
        state = "已暂停" if not group.enabled else "运行中"
        verdict = "够格收屎" if confidence >= threshold else "样本不足，暂不收屎"
        return CommandResult(
            f"🧻 {group.name or group_key}\n"
            f"方向：{group.direction.label()}｜{state}\n"
            f"可信度：{confidence:.2f}（阈值 {threshold:.2f}）→ {verdict}\n"
            f"原生屎：{group.native_message_count} 条 / {group.native_char_count} 字\n"
            f"采纳：{group.accepted_count}｜拒绝：{group.rejected_count}\n"
            f"口味：{describe_intake(group)}"
        )

    def list_groups(self) -> CommandResult:
        groups = self.storage.list_groups()
        if not groups:
            return CommandResult("还没有任何群注册。")
        lines = [f"共 {len(groups)} 个群："]
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
                f"· {group.name or group.group_key}｜{group.direction.label()}｜"
                f"可信度 {confidence:.2f}｜原生 {group.native_message_count} 条"
            )
        return CommandResult("\n".join(lines))

    def set_threshold(
        self, event: Any, args: list[str], *, is_admin: bool = False
    ) -> CommandResult:
        group_key = unified_origin(event)
        group = self.storage.get_group(group_key) if group_key else None
        if group is None:
            return CommandResult("本群还没注册。", ok=False)
        if not args:
            current = self.settings.threshold_for(group.threshold)
            return CommandResult(f"本群当前阈值：{current:.2f}")
        try:
            value = self.settings.parse_threshold(args[0])
        except ValueError:
            return CommandResult("阈值需要是 0-1 的小数或百分比。", ok=False)
        group.threshold = value
        self.storage.upsert_group(group)
        effective = self.settings.threshold_for(value)
        return CommandResult(f"✅ 本群阈值已设为 {effective:.2f}")

    def toggle(self, event: Any, *, enabled: bool, is_admin: bool = False) -> CommandResult:
        group_key = unified_origin(event)
        group = self.storage.get_group(group_key) if group_key else None
        if group is None:
            return CommandResult("本群还没注册。", ok=False)
        group.enabled = enabled
        self.storage.upsert_group(group)
        return CommandResult("▶️ 已恢复本群收发。" if enabled else "⏸️ 已暂停本群收发。")


# ---------------------------------------------------------------------- 工具
def unified_origin(event: Any) -> str:
    return str(getattr(event, "unified_msg_origin", "") or "")


def group_name(event: Any) -> str:
    obj = getattr(event, "message_obj", None)
    name = getattr(obj, "group_name", "") or ""
    if not name:
        group_id = getattr(event, "get_group_id", lambda: "")()
        name = f"群{group_id}" if group_id else "群"
    return str(name)[:60]


def is_group_event(event: Any) -> bool:
    getter = getattr(event, "get_group_id", None)
    if not callable(getter):
        return False
    return bool(getter())
