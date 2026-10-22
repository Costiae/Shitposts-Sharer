"""候选识别：把 AstrBot 的消息对象翻译成 Message，并判断“值得送审”。

这里是第一道闸门，只做很松的过滤（够长 / 带图 / 合并转发 / B站分享），
真正的“是不是屎”交给 LLM。宁可多送，不可漏送；但也别把每句“哈哈”都送出去。
"""

from __future__ import annotations

import re
from typing import Any, Iterable

from .bili import extract_bili_ids
from .config import Settings
from .forward import (
    fetch_forward_nodes,
    node_content,
    node_sender_name,
    onebot_segment_to_text,
)
from .models import Message, Segment

# B站：BV 号、b23 短链、视频页
_FORWARD_HINT = ("合并", "聊天记录", "转发")

_IMAGE_SCHEMES = ("http://", "https://", "file://")


def _component_type(comp: Any) -> str:
    """组件类型名。

    AstrBot 的组件用 ``type`` 属性（Plain / Image / Forward ...），但测试或
    其他平台对象可能没有，退回到类名。注意 ``Segment`` 自身没有 ``type``，
    直接 getattr 会得到 "segment" 这种无意义的类型名。
    """
    raw = getattr(comp, "type", None)
    if raw is None:
        raw = comp.__class__.__name__
    return str(raw).lower()


def segment_from_component(comp: Any, depth: int = 0) -> Segment | None:
    """把一个 AstrBot 消息组件转成 Segment；未知组件保留类型名。"""
    kind = _component_type(comp)

    if kind in ("plain", "text"):
        return Segment(kind="plain", text=str(getattr(comp, "text", "") or ""))

    if kind == "image":
        url = getattr(comp, "url", "") or getattr(comp, "file", "") or ""
        return Segment(kind="image", url=str(url))
    if kind == "record":
        return Segment(kind="record", url=str(getattr(comp, "file", "") or ""))

    if kind == "at":
        return Segment(kind="at", text=str(getattr(comp, "name", "") or getattr(comp, "qq", "") or ""))

    if kind == "face":
        return Segment(kind="face", text=str(getattr(comp, "id", "") or ""))

    if kind in ("reply", "quote"):
        # 引用：保留被引用文本（如果平台给了），方便 LLM 理解“在回谁”
        text = getattr(comp, "text", "") or getattr(comp, "content", "") or ""
        return Segment(kind="reply", text=str(text))

    if kind in ("file", "video"):
        return Segment(kind=kind, url=str(getattr(comp, "url", "") or getattr(comp, "file", "") or ""))

    if kind in ("forward", "nodes", "node") or hasattr(comp, "node_list") or hasattr(comp, "nodes"):
        # Forward 组件只有 id；存在 url 字段里，供 expand_forward 回源取正文
        forward_id = str(getattr(comp, "id", "") or getattr(comp, "res_id", "") or "")
        return Segment(
            kind="forward",
            url=forward_id,
            children=_forward_children(comp, depth),
        )

    if kind in ("json", "miniapp", "xml"):
        data = getattr(comp, "data", "") or ""
        return Segment(kind=kind, text=data if isinstance(data, str) else str(data))

    text = getattr(comp, "text", "") or ""
    return Segment(kind=kind, text=str(text))


def _forward_children(comp: Any, depth: int) -> list[Segment]:
    """从组件的 node_list/nodes 里取子节点。

    NOTE: AstrBot 的 ``Forward`` 组件只有 ``id``，从不带 node_list ——
    未展开的转发请走 ``expand_forward``。
    """
    if depth > 2:  # 防御嵌套爆炸
        return []
    nodes = getattr(comp, "node_list", None) or getattr(comp, "nodes", None) or []
    children: list[Segment] = []
    for node in nodes:
        name = (
            getattr(node, "sender_name", "")
            or getattr(node, "name", "")
            or getattr(node, "nickname", "")
            or "群友"
        )
        content = getattr(node, "content", None) or []
        if content and not isinstance(content, (list, tuple)):
            content = [content]
        person = Segment(kind="plain", text=f"[{name}]")
        children.append(person)
        for sub in content or []:
            if sub is None:
                continue
            segment = segment_from_component(sub, depth + 1)
            if segment is not None:
                children.append(segment)
    return children


def _segments_from_onebot_nodes(nodes: list[Any]) -> list[Segment]:
    """把 get_forward_msg 的节点列表转成 Segment。"""
    children: list[Segment] = []
    for node in nodes:
        name = node_sender_name(node) or "群友"
        children.append(Segment(kind="plain", text=f"[{name}]"))
        for piece in node_content(node):
            text = onebot_segment_to_text(piece)
            if text:
                children.append(Segment(kind="plain", text=text))
    return children


def message_from_event(event: Any, group_key: str) -> Message:
    """把 AstrMessageEvent 翻译成 Message（不展开转发，见 expand_forward）。"""
    obj = getattr(event, "message_obj", None)
    components = getattr(obj, "message", None) or []

    segments: list[Segment] = []
    for comp in components:
        if comp is None:
            continue
        segment = segment_from_component(comp)
        if segment is not None:
            segments.append(segment)

    sender_id = str(getattr(event, "get_sender_id", lambda: "")() or "")
    sender_name = str(getattr(obj, "sender_name", "") or "") or "群友"
    message_id = str(getattr(obj, "message_id", "") or "")
    return Message(
        group_key=group_key,
        sender_id=sender_id,
        sender_name=sender_name,
        segments=segments,
        message_id=message_id,
        ts=float(getattr(obj, "time", 0) or 0) or __import__("time").time(),
        raw=event,
    )


async def expand_forward(message: Message, context: Any) -> Message:
    """把消息里每个空壳 Forward 换成真实内容。

    AstrBot 的 Forward 组件只有 id，napcat 默认 parseMultMsg=false 时
    它是彻底的空的。这里调 get_forward_msg 补齐，避免要求用户改
    napcat 全局配置（那会影响所有插件，还可能要求重新扫码登录）。

    取不到就原样返回，上层仍按「空转发」处理，不会崩。
    """
    pending = [
        seg
        for seg in message.segments
        if seg.kind == "forward" and not seg.children
    ]
    if not pending:
        return message

    for segment in pending:
        forward_id = str(getattr(segment, "url", "") or "")
        if not forward_id:
            continue
        nodes = await fetch_forward_nodes(context, forward_id)
        if not nodes:
            continue
        children = _segments_from_onebot_nodes(nodes)
        if children:
            segment.children = children
    return message


def is_there_a_shit(message: Message, settings: Settings) -> tuple[bool, str]:
    """粗筛：返回 (是否送审, 原因)。规则刻意宽松。"""
    if not settings.get("enabled"):
        return False, "插件已关闭"

    kinds = message.kinds()
    text = message.text()

    if "forward" in kinds:
        return True, "合并转发"

    if "image" in kinds:
        return True, "图片"

    bili = extract_bili_ids(text)
    if bili.bvids or bili.short_urls or bili.shares:
        return True, "B站分享"

    if "json" in kinds or "miniapp" in kinds or "xml" in kinds:
        # 卡片消息（分享）多半带链接，交给 LLM 看
        return True, "卡片分享"

    if message.char_count() >= settings.text_min_chars:
        return True, "长文本"

    return False, ""


def summarise_segments(segments: Iterable[Segment], limit: int = 400) -> str:
    """给日志/提示词用的简短预览。"""
    parts = []
    for seg in segments:
        text = seg.plain_text()
        if text:
            parts.append(text)
    joined = "\n".join(parts)
    return joined[:limit] + ("…" if len(joined) > limit else "")


def looks_like_image_only(message: Message) -> bool:
    kinds = message.kinds()
    return kinds == {"image"} or kinds == {"image", "at"}


def has_url(text: str) -> bool:
    return bool(re.search(r"https?://", text or ""))
