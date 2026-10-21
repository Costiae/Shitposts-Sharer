"""上下文窗口：把“一段时间内的群聊”攒成一坨，判定它们之间有没有上下文关系。

需求原话：“第二个要判断这些屎有无上下文关系。”

做法：
- 每个群一个滑动窗口，窗口内保留最近若干条消息（含未命中的）；
- 命中候选时开启一个“收集窗口”，把随后的消息也吸进来，最多 ``window_seconds``；
- 到点（或攒满条数）统一交给 LLM，由它判断候选之间是不是同一个话题、
  群友反应是不是同一件事。

不追求精确：这是“感觉一下是不是在聊同一件事”，不是聚类算法。
"""

from __future__ import annotations

import time
from collections import deque
from dataclasses import dataclass, field

from .config import Settings
from .models import ContextMessage, Message


@dataclass(slots=True)
class Suspicion:
    """一次“疑似屎”的现场。"""

    message: Message
    reason: str
    ts: float = field(default_factory=time.time)
    context_before: list[ContextMessage] = field(default_factory=list)
    extra_messages: list[Message] = field(default_factory=list)
    closed: bool = False

    def all_messages(self) -> list[Message]:
        return [self.message, *self.extra_messages]

    def age(self, now: float | None = None) -> float:
        return (now or time.time()) - self.ts


class ContextAccumulator:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        # group_key -> 最近消息（用于“前文”）
        self._history: dict[str, deque[ContextMessage]] = {}
        # group_key -> 正在积累的现场
        self._open: dict[str, Suspicion] = {}

    # ---------------------------------------------------------------- 输入
    def observe(
        self,
        message: Message,
        *,
        is_shit: bool = False,
        has_bili: bool = False,
    ) -> None:
        """所有群消息都过一遍：一是做前文，二是喂给已打开的窗口。"""
        entry = ContextMessage(
            sender_name=message.sender_name,
            text=message.text()[:800],
            image_count=sum(1 for seg in message.segments if seg.kind == "image"),
            is_shit=is_shit,
            has_bili=has_bili,
            ts=message.ts,
        )

        history = self._history.setdefault(
            message.group_key, deque(maxlen=self.settings.get("history_max_messages"))
        )
        history.append(entry)

        suspicion = self._open.get(message.group_key)
        if suspicion is not None and not suspicion.closed:
            if message is not suspicion.message:
                suspicion.extra_messages.append(message)
                if len(suspicion.extra_messages) >= self.settings.max_messages_per_shit:
                    suspicion.closed = True

    def open(self, message: Message, reason: str) -> Suspicion:
        """命中候选，开一个现场，并把之前的若干条消息当“前文”带上。"""
        history = list(self._history.get(message.group_key, []))
        # 去掉当前这条本身（它已经在 observe 里进 history 了）
        if history and history[-1].text == message.text()[:800]:
            history = history[:-1]
        before = history[-self.settings.get("history_max_messages") :]

        suspicion = Suspicion(
            message=message,
            reason=reason,
            context_before=[item for item in before],
        )
        self._open[message.group_key] = suspicion
        return suspicion

    # ---------------------------------------------------------------- 收盘
    def due(self, now: float | None = None) -> list[tuple[str, Suspicion]]:
        """返回已经到点、可以送去分析的现场。"""
        moment = now or time.time()
        ready: list[tuple[str, Suspicion]] = []
        for group_key, suspicion in list(self._open.items()):
            if suspicion.closed or suspicion.age(moment) >= self.settings.window_seconds:
                ready.append((group_key, suspicion))
                self._open.pop(group_key, None)
        return ready

    def pending(self) -> int:
        return len(self._open)

    def drop(self, group_key: str) -> None:
        self._open.pop(group_key, None)

    # ---------------------------------------------------------------- 上下文
    def render_context(self, suspicion: Suspicion, limit: int | None = None) -> str:
        """把前文 + 后续消息压成给 LLM 读的文本。"""
        before = suspicion.context_before
        if limit:
            before = before[-limit:]
        lines = []
        for item in before:
            marks = []
            if item.image_count:
                marks.append(f"{item.image_count}张图")
            if item.has_bili:
                marks.append("B站")
            suffix = f"（{'，'.join(marks)}）" if marks else ""
            lines.append(f"{item.sender_name}：{item.text}{suffix}")
        return "\n".join(lines)

    def render_extra(self, suspicion: Suspicion) -> str:
        """候选之后群里接着说了什么。"""
        lines = []
        for message in suspicion.extra_messages[-self.settings.max_messages_per_shit :]:
            lines.append(f"{message.sender_name}：{message.text()[:400]}")
        return "\n".join(lines)
