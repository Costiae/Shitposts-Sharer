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

# 链接分享：只认真正的 http(s) URL，避免把 "http" 这种词误判成链接
_LINK_RE = re.compile(r"https?://[^\s<>\"']+", re.IGNORECASE)

# “玩梗短句”标记：这些词单独出现不值得判定，但配合短句往往是抽象现场。
# 只作为粗筛入口，真正的判断仍然交给 LLM，所以宁可松一点。
_LOOSE_MARKER_RE = re.compile(
    r"(哈{3,}|草{2,}|6{3,}|笑死|绷不住|绷|逆天|离谱|抽象|裂开|典中典|急了|乐死|卧槽)"
)

# 媒体类载体：语音 / 视频 / 文件本身就是“值得看一眼”的内容
_MEDIA_KINDS = ("record", "video", "file")


def _component_type(comp: Any) -> str:
    """组件类型名，统一成小写字符串。

    AstrBot 4.x 的 ``ComponentType`` 是 Enum：在非 StrEnum 的版本里，
    ``str(ComponentType.Plain)`` 是 ``"ComponentType.Plain"``，直接 lower
    会得到 ``"componenttype.plain"``，于是下面所有分支都匹配不上 ——
    图片、转发、卡片会全部退化成 unknown，表现为“识别不出任何东西”。

    所以必须优先取枚举的 ``.value``；同时兼容已经是字符串的旧对象，
    以及某些版本把类型名写成 ``"ComponentType.Plain"`` 的形式。
    """
    raw = getattr(comp, "type", None)
    if raw is None:
        raw = comp.__class__.__name__

    value = getattr(raw, "value", raw)
    name = str(value)
    lowered = name.lower()
    if "." in name and lowered.startswith("componenttype."):
        name = name.rsplit(".", 1)[-1]
    return name.lower()


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
    """粗筛：返回 (是否送审, 原因)。规则刻意宽松。

    判定顺序就是“证据强度”顺序：明确载体（转发/图片/媒体/卡片）→ 平台分享
    （B站/链接）→ 文本长度 → 玩梗信号。越靠前的规则越不需要模型兜底，
    所以原因字符串里会带上具体证据（几张图、几条转发、命中哪个梗），
    方便在 ``[shit] detect`` 日志里直接看出为什么被俘获。
    """
    if not settings.get("enabled"):
        return False, "插件已关闭"

    kinds = message.kinds()
    text = message.text()
    chars = message.char_count()

    if "forward" in kinds:
        nodes = sum(len(seg.children) for seg in message.segments if seg.kind == "forward")
        need = settings.forward_min_nodes
        if nodes and nodes < need:
            return False, f"转发只有 {nodes} 条 < {need}"
        detail = f"{nodes}条" if nodes else "空壳"
        return True, f"合并转发({detail})"

    if "image" in kinds:
        images = sum(1 for seg in message.segments if seg.kind == "image")
        return True, f"图片({images}张)"

    if settings.detect_media:
        media = [kind for kind in kinds if kind in _MEDIA_KINDS]
        if media:
            return True, "媒体(" + "/".join(sorted(media)) + ")"

    bili = extract_bili_ids(text)
    if bili.bvids or bili.short_urls or bili.shares:
        return True, "B站分享"

    if "json" in kinds or "miniapp" in kinds or "xml" in kinds:
        # 卡片消息（分享）多半带链接，交给 LLM 看
        return True, "卡片分享"

    if settings.detect_links and _LINK_RE.search(text):
        return True, "链接分享"

    if chars >= settings.text_min_chars:
        return True, f"长文本({chars}字)"

    if not settings.loose_detect:
        return False, ""

    # ---- 松闸门：宁可多送一次模型，也别漏掉真正的抽象现场 ----
    # 1) 中等长度的文本：常常是“一段没头没尾的抽象故事”
    if chars >= settings.soft_text_min_chars:
        return True, f"中等文本({chars}字)"

    # 2) 短句里的梗信号：哈哈哈哈哈 / 草 / 逆天 / 绷不住 ...
    marker = _LOOSE_MARKER_RE.search(text)
    if marker:
        return True, f"玩梗短句({marker.group(0)})"

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
