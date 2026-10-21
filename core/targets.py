"""指令目标群解析：让管理员在 A 群操作 B 群。

搬屎的接收方是静默的 —— 群里的人不会、也不该去配置"自己这个群怎么收屎"。
真正操作的人通常坐在一个**自己的群**里，要去管别的群。所以指令的第一段参数
永远是"目标群"，省略时默认就是当前会话。

支持的目标写法：

| 写法 | 例子 | 说明 |
| --- | --- | --- |
| 省略 | `/shit register in` | 当前会话 |
| 纯数字 | `/shit status 123456` | 会话 ID，自动匹配已注册群的 ``<platform>:<type>:<id>`` |
| 完整 UMO | `/shit status aiocqhttp:GroupMessage:123456` | AstrBot 的 unified_msg_origin |
| 已注册群序号 | `/shit status #2` | 配合 `/shit list` 使用 |

解析结果分三种：命中唯一群、命中多个（需消歧）、找不到。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .models import GroupProfile
from .storage import Storage


@dataclass(slots=True)
class TargetResolution:
    status: str  # ok | ambiguous | not_found | empty
    group: GroupProfile | None = None
    candidates: list[GroupProfile] | None = None
    raw: str = ""
    note: str = ""

    @property
    def ok(self) -> bool:
        return self.status == "ok" and self.group is not None


def current_origin(event: Any) -> str:
    return str(getattr(event, "unified_msg_origin", "") or "")


def looks_like_target(token: str) -> bool:
    """判断一个参数是不是"目标群"而不是子指令/阈值。

    纯数字、带冒号的 UMO、``#n`` 序号都算目标；``in``/``both``/``0.8``/``80%`` 不算。
    """
    if not token:
        return False
    stripped = token.lstrip("#")
    if token.startswith("#") and stripped.isdigit():
        return True
    if ":" in token:
        return True
    return token.isdigit()


def _session_id(origin: str) -> str:
    return origin.rsplit(":", 1)[-1] if origin else ""


def _platform(origin: str) -> str:
    return origin.split(":", 1)[0] if origin else ""


def resolve(
    storage: Storage,
    event: Any,
    token: str = "",
    *,
    want_unregistered: bool = False,
) -> TargetResolution:
    """把指令里的目标参数解析成具体群。

    ``want_unregistered=True`` 时（例如 register），允许目标群尚未注册。
    """
    if not token:
        origin = current_origin(event)
        if not origin:
            return TargetResolution(status="empty", note="no current session")
        group = storage.get_group(origin)
        if group is None:
            if not want_unregistered:
                return TargetResolution(
                    status="not_found",
                    raw=origin,
                    note="this session is not registered",
                )
            # 用当前会话的显示名占位，注册时再补
            group = GroupProfile(
                group_key=origin, name=_session_name(event)
            )
        return TargetResolution(status="ok", group=group, raw=origin)

    raw = token.strip()
    registered = storage.list_groups()
    by_origin = {group.group_key: group for group in registered}

    # 1) 完整 UMO 精确命中
    if ":" in raw:
        group = by_origin.get(raw)
        if group is not None:
            return TargetResolution(status="ok", group=group, raw=raw)
        if want_unregistered:
            return TargetResolution(
                status="ok",
                group=GroupProfile(group_key=raw, name=_short(raw)),
                raw=raw,
                note="not registered yet",
            )
        return TargetResolution(status="not_found", raw=raw)

    # 2) #n 序号（对齐 /shit list 的打印顺序）
    if raw.startswith("#") and raw[1:].isdigit():
        index = int(raw[1:])
        if 1 <= index <= len(registered):
            return TargetResolution(status="ok", group=registered[index - 1], raw=raw)
        return TargetResolution(status="not_found", raw=raw, note="index out of range")

    # 3) 纯数字：按会话 ID 匹配，可能跨平台命中多个
    if raw.isdigit():
        hits = [group for group in registered if _session_id(group.group_key) == raw]
        if len(hits) == 1:
            return TargetResolution(status="ok", group=hits[0], raw=raw)
        if len(hits) > 1:
            return TargetResolution(
                status="ambiguous", candidates=hits, raw=raw, note="multiple platforms"
            )
        if want_unregistered:
            origin = current_origin(event)
            platform = _platform(origin) or "aiocqhttp"
            guess = f"{platform}:GroupMessage:{raw}"
            return TargetResolution(
                status="ok",
                group=GroupProfile(group_key=guess, name=raw),
                raw=raw,
                note="guessed origin",
            )
        return TargetResolution(status="not_found", raw=raw)

    # 4) 群名模糊匹配
    lowered = raw.lower()
    hits = [group for group in registered if lowered in (group.name or "").lower()]
    if len(hits) == 1:
        return TargetResolution(status="ok", group=hits[0], raw=raw)
    if len(hits) > 1:
        return TargetResolution(status="ambiguous", candidates=hits, raw=raw)
    return TargetResolution(status="not_found", raw=raw)


def describe_group(group: GroupProfile) -> str:
    """给指令回复用的短标识，永远能反查回去。"""
    return f"{group.name or _session_id(group.group_key)} ({group.group_key})"


def render_ambiguous(resolution: TargetResolution, storage: Storage) -> str:
    lines = [f"Ambiguous target '{resolution.raw}', pick one:"]
    registered = storage.list_groups()
    for group in resolution.candidates or []:
        try:
            index = registered.index(group) + 1
        except ValueError:
            index = "?"
        lines.append(f"  #{index}  {group.group_key}")
    return "\n".join(lines)


def _session_name(event: Any) -> str:
    obj = getattr(event, "message_obj", None)
    name = getattr(obj, "group_name", "") or ""
    if not name:
        getter = getattr(event, "get_group_id", None)
        group_id = getter() if callable(getter) else ""
        name = str(group_id or "")
    return str(name)[:60]


def _short(origin: str) -> str:
    return _session_id(origin) or origin
