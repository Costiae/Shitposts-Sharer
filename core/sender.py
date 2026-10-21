"""把一条屎重新组装成消息，发到目标群。

为什么不用 ``event.send``：那只能发回“当前群”。跨群必须重新构造
``AstrMessageEvent`` —— 用 ``MessageSession`` / ``send_message``
（以平台适配器的上下文为准），再把原消息的组件原样塞进去。

图片有两种情况：
- URL 还在（http/file）：直接复刻 Image 组件；
- 平台要求本地文件或图片已失效：退化为发送 URL 文本，不阻塞整条搬运。
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import Any

from .models import Segment, ShitRecord

_LABEL = "【搬屎】"
_MAX_RETRY = 2


@dataclass(slots=True)
class DeliveryResult:
    group_key: str
    ok: bool
    detail: str = ""

    def as_status(self) -> str:
        return "ok" if self.ok else "fail"


def _plain(text: str):
    from astrbot.api.message_components import Plain  # type: ignore[import-not-found]

    return Plain(text)


def _image(url: str):
    from astrbot.api.message_components import Image  # type: ignore[import-not-found]

    return Image.fromURL(url) if url.startswith(("http", "file")) else Image(file=url)


def _forward(payload_children: list[Any]):
    from astrbot.api.message_components import Nodes  # type: ignore[import-not-found]

    return Nodes(nodes=payload_children)


def build_chain(
    record: ShitRecord,
    *,
    label: str = _LABEL,
    source_name: str = "",
    with_context: bool = True,
    context_text: str = "",
) -> list[Any]:
    """把 ShitRecord 变成一个可发送的消息链（组件列表）。"""
    chain: list[Any] = []

    header = label or _LABEL
    if source_name:
        header = f"{header} 来自「{source_name}」"
    chain.append(_plain(header + "\n"))

    for item in record.payload:
        segment = Segment.from_dict(item)
        chain.extend(_segments_to_component(segment))

    if with_context and record.context_text:
        chain.append(_plain("\n—— 搬的时候群里在聊 ——\n" + record.context_text))

    if record.reason:
        chain.append(_plain(f"\n（{record.reason}）"))

    return chain


def _segments_to_component(segment: Segment) -> list[Any]:
    if segment.kind == "plain":
        return [_plain(segment.text)] if segment.text else []
    if segment.kind == "image":
        if not segment.url:
            return [_plain("[图片]")]
        return [_image(segment.url)]
    if segment.kind in ("record", "video", "file"):
        if segment.url:
            return [_plain(segment.url)]
        return []
    if segment.kind in ("json", "miniapp", "xml"):
        # 卡片消息一般无法跨群复刻，退化为文本提示
        return [_plain("[分享卡片]")]
    if segment.kind == "forward":
        children = []
        for child in segment.children:
            children.extend(_segments_to_component(child))
        if not children:
            return [_plain("[空转发]")]
        return [_forward(children)]
    if segment.text:
        return [_plain(segment.text)]
    return []


async def send_to_group(context: Any, group_key: str, chain: list[Any]) -> DeliveryResult:
    """用平台适配器把消息发到指定会话。"""
    # 新版 AstrBot：context.send_message(umo, message)
    sender = getattr(context, "send_message", None)
    if sender is None:
        return DeliveryResult(
            group_key=group_key,
            ok=False,
            detail="当前 AstrBot 版本没有可用的跨群发送接口",
        )

    last_error = ""
    for attempt in range(1 + _MAX_RETRY):
        try:
            result = sender(group_key, chain)
            if hasattr(result, "__await__"):
                await result
            return DeliveryResult(group_key=group_key, ok=True)
        except Exception as exc:  # pragma: no cover - 网络类异常
            last_error = f"{type(exc).__name__}: {exc}"
            if attempt < _MAX_RETRY:
                await asyncio.sleep(1.5 * (attempt + 1))

    return DeliveryResult(group_key=group_key, ok=False, detail=last_error)
