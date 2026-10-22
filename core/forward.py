"""通过 OneBot API 展开合并转发消息。

为什么需要这一层：

AstrBot 的 ``Forward`` 组件只有一个 ``id`` 字段（见
``astrbot/core/message/components.py``），**不含任何正文**：

    class Forward(BaseMessageComponent):
        type: ComponentType = ComponentType.Forward
        id: str

``Node`` 类型存在，但从不会出现在 Forward 里。napcat 默认
``parseMultMsg = false``，因此合并转发到达插件时就是一个空壳 —— 模型只能
看到「（无文本内容，只有媒体）」，于是无论内容多逆天都判成 is_shit=false。

这里不要求用户去改 napcat 配置（那会影响所有插件，且可能要重新扫码），
而是自己调 ``get_forward_msg`` 把正文取回来。

参考实现：AstrBot 自己在处理 ``reply`` 段时用的就是同一套
``bot.call_action(...)`` 路径（见 aiocqhttp_platform_adapter.py）。
"""

from __future__ import annotations

import asyncio
from typing import Any

# napcat / go-cqhttp 的两种返回形状都要认
_NODE_LIST_KEYS = ("messages", "message", "content", "nodes", "node_list")


def get_onebot_client(context: Any) -> Any | None:
    """从 AstrBot context 里挖出 OneBot/CQHttp client。

    路径：context.platform_manager.platform_insts[*].get_client()
    只有 aiocqhttp（NapCat / LLOneBot / Lagrange）适配器才有这个方法。
    """
    manager = getattr(context, "platform_manager", None)
    if manager is None:
        return None
    for platform in getattr(manager, "platform_insts", None) or []:
        getter = getattr(platform, "get_client", None)
        if not callable(getter):
            continue
        try:
            client = getter()
        except Exception:
            continue
        if client is not None and hasattr(client, "call_action"):
            return client
    return None


def _extract_nodes(payload: Any) -> list[Any]:
    """从 get_forward_msg 的返回里取出节点列表。"""
    if payload is None:
        return []
    if isinstance(payload, list):
        return payload
    if isinstance(payload, dict):
        for key in _NODE_LIST_KEYS:
            value = payload.get(key)
            if isinstance(value, list):
                return value
        data = payload.get("data")
        if isinstance(data, dict):
            return _extract_nodes(data)
    return []


async def fetch_forward_nodes(
    context: Any, forward_id: str, *, timeout: float = 8.0
) -> list[Any]:
    """取回一条合并转发的节点列表；失败返回空列表，绝不抛。

    失败原因（id 过期、权限不足、超时、非 OneBot 平台）都不该让插件崩掉，
    上层会退回「只看到空壳」的老行为。
    """
    if not forward_id:
        return []
    client = get_onebot_client(context)
    if client is None:
        return []
    try:
        payload = await asyncio.wait_for(
            client.call_action("get_forward_msg", id=str(forward_id)),
            timeout=timeout,
        )
    except asyncio.TimeoutError:
        return []
    except Exception:
        return []
    return _extract_nodes(payload)


def _node_body(node: Any) -> dict[str, Any]:
    """取出节点的真实字段。

    napcat 的 get_forward_msg 返回的是 OneBot 段包装：

        {"type": "node", "data": {"sender": {...}, "message": [...]}}

    有些实现直接把字段平铺在顶层，所以两种都认。
    """
    if not isinstance(node, dict):
        return {}
    data = node.get("data")
    if isinstance(data, dict) and any(
        key in data for key in ("sender", "message", "content", "nickname", "name")
    ):
        return data
    return node


def node_sender_name(node: Any) -> str:
    """节点发送者昵称：napcat 放在 sender 里，有的实现直接平铺。"""
    body = _node_body(node)
    sender = body.get("sender")
    if isinstance(sender, dict):
        for key in ("card", "nickname", "name"):
            value = sender.get(key)
            if value:
                return str(value)
    for key in ("sender_name", "nickname", "name", "uin"):
        value = body.get(key)
        if value:
            return str(value)
    return ""


def node_content(node: Any) -> list[dict[str, Any]]:
    """节点正文：标准 OneBot 是 {"data": {...}, "type": "text"} 的列表。"""
    body = _node_body(node)
    raw = body.get("content") or body.get("message")
    if isinstance(raw, list):
        return [item for item in raw if isinstance(item, dict)]
    if isinstance(raw, dict):
        return [raw]
    return []


def onebot_segment_to_text(seg: dict[str, Any]) -> str:
    """把一条 OneBot 消息段渲染成可读文本。"""
    seg_type = str(seg.get("type") or "").lower()
    data = seg.get("data")
    if not isinstance(data, dict):
        return ""
    if seg_type == "text":
        return str(data.get("text") or "")
    if seg_type == "image":
        return "[图片]"
    if seg_type == "face":
        return "[表情]"
    if seg_type == "record":
        return "[语音]"
    if seg_type == "video":
        return "[视频]"
    if seg_type == "at":
        return f"@{data.get('name') or data.get('qq') or ''}"
    if seg_type == "json":
        return str(data.get("data") or "")[:500]
    if seg_type == "file":
        return f"[文件:{data.get('name') or ''}]"
    return ""
