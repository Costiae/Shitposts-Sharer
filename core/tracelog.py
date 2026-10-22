"""链路追踪：把每条消息的每一道闸门都写进日志。

设计目标（来自需求）：
1. 群里的**每一条**消息都要留痕 —— 包括被上游 AstrBot 过滤掉、插件压根没
   收到的那些（见 main.py 的 on_any_message 里的 EARLY 日志）；
2. 每一层都要说明**是否俘获**（caught）以及**为什么没俘获**（why not）；
3. 分析过程要能看到模型拿到了什么、回了什么。

日志量很大，所以有三档：

    off    只写最终结果（搬运成功/失败）
    normal 每个可疑消息一条汇总（默认）
    full   每一层每一步都写，含提示词摘要与原始模型输出

用 ``log_level`` 配置控制。所有行前缀 ``[shit]`` 便于 grep。
"""

from __future__ import annotations

from typing import Any

from .models import Message

LEVELS = ("off", "normal", "full")


def _clip(text: str, limit: int) -> str:
    text = " ".join(str(text).split())
    if len(text) <= limit:
        return text
    return text[: limit - 1] + "…"


class Trace:
    """一条消息的追踪上下文。

    ``caught=False`` 时每一次 ``skip()`` 都会记录被哪道闸门拦下，
    这样「为什么没搬」永远有答案。
    """

    __slots__ = ("logger", "level", "stage", "trail", "was_caught")

    def __init__(self, logger: Any, level: str = "normal") -> None:
        self.logger = logger
        self.level = level if level in LEVELS else "normal"
        self.stage = "received"
        self.trail: list[str] = []
        self.was_caught = False

    # ------------------------------------------------------------ 内部
    def _enabled(self) -> bool:
        return self.level != "off"

    def _write(self, level: str, key: str, text: str) -> None:
        if not self._enabled():
            return
        line = f"[shit] {key} | {text}"
        if level == "debug":
            self.logger.debug(line)
        elif level == "warning":
            self.logger.warning(line)
        elif level == "error":
            self.logger.error(line)
        else:
            self.logger.info(line)

    # ------------------------------------------------------------ 阶段
    def gate(self, name: str) -> None:
        """标记当前阶段（用于 skip 时说明卡在哪）。"""
        self.stage = name

    def skip(self, reason: str) -> None:
        """记录一道闸门没通过。这是「没俘获」的主要来源。

        normal 及以上都写：如果只在 full 下写，"这条消息为什么没被搬"
        在默认配置里就是一个黑洞。
        """
        self.trail.append(f"{self.stage}: {reason}")
        # normal 也要能直接看到“为什么没被俘获”；debug 级日志在默认
        # AstrBot 日志级别下通常不可见，会让 log_level=normal 名不副实。
        self._write("info", self.stage, f"SKIP {reason}")

    def caught(self, reason: str) -> None:
        """标记为潜在屎，进入分析。

        trail 里只留简短原因：它在 full 级别会被整条打印，塞进消息正文
        会让一行日志长得没法看。正文已经在 CAUGHT 那行上下文里。
        """
        self.was_caught = True
        short = reason.split("|")[0].strip()
        self.trail.append(f"{self.stage}: CAUGHT {short}")
        self._write("info", self.stage, f"CAUGHT {reason}")

    def step(self, text: str) -> None:
        """单步进展，只在 full 下输出。"""
        if self.level == "full":
            self._write("info", self.stage, text)

    def result(self, text: str) -> None:
        """最终结论，off 之外都写。"""
        self._write("info", self.stage, text)

    def warn(self, text: str) -> None:
        self._write("warning", self.stage, text)

    def error(self, text: str) -> None:
        self._write("error", self.stage, text)

    def summary(self) -> str:
        """把整条链压成一行，用于消息结束时统一输出。"""
        return " → ".join(self.trail) if self.trail else "no events"


def message_key(message: Message) -> str:
    """日志里的消息标识：群 + 发送者，短但可辨。"""
    group = message.group_key.rsplit(":", 1)[-1]
    who = message.sender_name or message.sender_id or "?"
    return f"{group}/{who}"


def describe_message(message: Message, *, limit: int = 240) -> str:
    """把一条消息渲染成适合进日志的一行摘要。"""
    kinds = "+".join(sorted(message.kinds())) or "empty"
    text = _clip(message.text(), limit)
    parts = [f"kinds={kinds}"]
    if message.sender_name:
        parts.append(f"by={message.sender_name}")
    if message.message_id:
        parts.append(f"msg={message.message_id}")
    parts.append(f"text={text!r}" if text else "text=<empty>")
    return " ".join(parts)


def describe_segments(message: Message, *, limit: int = 400, depth: int = 0) -> str:
    """逐段展开，full 级别用。合并转发会递归展示子节点。"""
    out = []
    indent = "  " * depth
    for index, seg in enumerate(message.segments):
        piece = f"{indent}#{index}:{seg.kind}"
        if seg.text:
            piece += f"({_clip(seg.text, 60)!r})"
        if seg.url:
            piece += f"<{_clip(seg.url, 80)}>"
        if seg.children:
            piece += f"[{len(seg.children)} children]"
        out.append(piece)
        if seg.children and depth < 2:
            nested = describe_children(seg.children, limit=limit, depth=depth + 1)
            if nested:
                out.append(nested)
    return "\n".join(out) if out else "<no segments>"


def describe_children(children: list[Any], *, limit: int = 400, depth: int = 0) -> str:
    """展开转发内部的子节点，带缩进。"""
    indent = "  " * depth
    out = []
    for index, child in enumerate(children[:limit]):
        text = getattr(child, "text", "") or ""
        kind = getattr(child, "kind", "?")
        if text:
            out.append(f"{indent}- [{index}] {kind}: {_clip(text, 70)!r}")
        else:
            out.append(f"{indent}- [{index}] {kind}")
    if len(children) > limit:
        out.append(f"{indent}- ... {len(children) - limit} more")
    return "\n".join(out)
