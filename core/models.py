"""核心数据结构。

这里只放“数据”与“数据上的纯函数”，不碰 IO、不碰 AstrBot、不碰 LLM，
方便在单元测试里直接构造。
"""

from __future__ import annotations

import hashlib
import math
import re
import time
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any


class Direction(StrEnum):
    """群在搬运拓扑中的角色。"""

    IN = "in"  # 只收
    OUT = "out"  # 只发
    BOTH = "both"  # 进出

    @property
    def can_receive(self) -> bool:
        return self in (Direction.IN, Direction.BOTH)

    @property
    def can_emit(self) -> bool:
        return self in (Direction.OUT, Direction.BOTH)

    @classmethod
    def parse(cls, raw: str) -> "Direction":
        value = (raw or "").strip().lower()
        # 中文别名：群友手滑也认
        alias = {
            "收": cls.IN,
            "入": cls.IN,
            "进": cls.IN,
            "只收": cls.IN,
            "出": cls.OUT,
            "发": cls.OUT,
            "只发": cls.OUT,
            "进出": cls.BOTH,
            "收发": cls.BOTH,
            "都": cls.BOTH,
        }
        if value in alias:
            return alias[value]
        for member in cls:
            if member.value == value:
                return member
        raise ValueError(f"未知的搬运方向: {raw!r}（可用 in / out / both）")

    def label(self) -> str:
        return {self.IN: "只收", self.OUT: "只发", self.BOTH: "进出"}[self]


class Kind(StrEnum):
    """屎的载体类型。"""

    TEXT = "text"
    IMAGE = "image"
    FORWARD = "forward"
    BILI = "bili"
    MIXED = "mixed"
    PROMOTED = "promoted"  # 由群友讨论累积出来的“话题屎”


@dataclass(slots=True)
class Segment:
    """一条原始消息片段的规范化形式，用于持久化与再次回放发送。"""

    kind: str  # plain | image | at | face | reply | file | json | video | other
    text: str = ""
    url: str = ""
    # 合并转发在此递归保存
    children: list["Segment"] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        data: dict[str, Any] = {"kind": self.kind}
        if self.text:
            data["text"] = self.text
        if self.url:
            data["url"] = self.url
        if self.children:
            data["children"] = [child.to_dict() for child in self.children]
        return data

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Segment":
        return cls(
            kind=str(data.get("kind", "other")),
            text=str(data.get("text", "")),
            url=str(data.get("url", "")),
            children=[cls.from_dict(child) for child in data.get("children", [])],
        )

    def plain_text(self, depth: int = 0) -> str:
        """把段落压成可读文本，供 LLM 阅读与哈希。"""
        pad = "  " * depth
        if self.kind == "plain":
            return pad + self.text
        if self.kind == "at":
            return pad + f"@{self.text}".rstrip("@")
        if self.kind == "face":
            return pad + f"[表情{self.text}]"
        if self.children:
            inner = "\n".join(
                child.plain_text(depth + 1) for child in self.children if child is not None
            )
            return f"{pad}[转发{self.text or ''}:{depth}]\n{inner}"
        return pad + f"[{self.kind}]{self.text}"


@dataclass(slots=True)
class Message:
    """一条群消息。"""

    group_key: str
    sender_id: str = ""
    sender_name: str = "群友"
    segments: list[Segment] = field(default_factory=list)
    ts: float = field(default_factory=time.time)
    message_id: str = ""
    raw: Any = None  # 原事件（回放发图需要原件），不参与序列化

    def text(self) -> str:
        return "\n".join(
            seg.plain_text() for seg in self.segments if seg is not None
        ).strip()

    def kinds(self) -> set[str]:
        found: set[str] = set()
        for seg in self.segments:
            found.add(seg.kind)
        return found

    def has_image(self) -> bool:
        return any(seg.kind == "image" for seg in self.segments)

    def has_forward(self) -> bool:
        return any(seg.kind == "forward" for seg in self.segments)

    def char_count(self) -> int:
        return len(re.sub(r"\s+", "", self.text()))

    def to_dict(self) -> dict[str, Any]:
        return {
            "kind": self.kind_hint(),
            "ts": self.ts,
            "text": self.text(),
            "sender_id": self.sender_id,
            "sender_name": self.sender_name,
            "segments": [seg.to_dict() for seg in self.segments],
        }

    def kind_hint(self) -> str:
        """粗分载体类型，用于表情/图片的兜底指纹。"""
        kinds = self.kinds()
        if "forward" in kinds:
            return "forward"
        if "image" in kinds:
            return "image"
        return "text"


@dataclass(slots=True)
class GroupProfile:
    """一个注册群的档案。"""

    group_key: str
    name: str = ""
    direction: Direction = Direction.BOTH
    intake: dict[str, float] = field(default_factory=dict)
    threshold: float = 0.0  # 单群覆盖阈值，0 表示用全局
    enabled: bool = True
    native_message_count: int = 0
    native_char_count: int = 0
    accepted_count: int = 0
    rejected_count: int = 0
    last_shit_at: float = 0.0
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)

    # ---- 可信度 ----
    def confidence(self, k: float = 10.0) -> float:
        """样本不足的可信度惩罚：n/(n+k)。

        样本量为 0 时可信度为 0（完全不可信），样本很多时趋于 1。
        惩罚得很温和，因为它只是防武断，不是精度核心。
        """
        n = max(0, self.native_message_count)
        return n / (n + max(k, 1e-6))


@dataclass(slots=True)
class ContextMessage:
    """参与上下文的一条消息（比 Message 轻，只留必要字段）。"""

    sender_name: str
    text: str
    image_count: int = 0
    is_shit: bool = False
    has_bili: bool = False
    ts: float = field(default_factory=time.time)


@dataclass(slots=True)
class ShitRecord:
    """一条被判定为屎的内容。"""

    shit_id: str
    source_group: str
    ts: float = field(default_factory=time.time)
    kind: Kind = Kind.TEXT
    text_blob: str = ""
    keywords: list[str] = field(default_factory=list)
    vector: dict[str, float] = field(default_factory=dict)
    fingerprint: str = ""
    incoming: bool = False
    tags: list[str] = field(default_factory=list)
    intensity: float = 0.0
    context_included: bool = False
    payload: list[dict[str, Any]] = field(default_factory=list)
    context_text: str = ""
    reason: str = ""


@dataclass(slots=True)
class Analysis:
    """LLM 对一条候选内容的判断结果。"""

    is_shit: bool = False
    intensity: float = 0.0
    tags: list[str] = field(default_factory=list)
    keywords: list[str] = field(default_factory=list)
    reason: str = ""
    context_relevant: bool = False
    context_keywords: list[str] = field(default_factory=list)
    topic_seed: str = ""


@dataclass(slots=True)
class RouteCandidate:
    """一个目标群的匹配结果。"""

    group_key: str
    score: float
    confidence: float
    source: str = "keyword"  # keyword | llm
    reason: str = ""

    @property
    def final_score(self) -> float:
        """排序分：匹配度与可信度相乘，简单且能防武断。"""
        return self.score * self.confidence


def sha1_text(text: str) -> str:
    return hashlib.sha1(text.encode("utf-8")).hexdigest()


def normalize_text(text: str) -> str:
    """压缩空白、去掉零宽字符，降低同内容不同排版的指纹漂移。"""
    cleaned = re.sub(r"[\u200b-\u200f\u2060\ufeff]", "", text or "")
    return re.sub(r"\s+", " ", cleaned).strip()


def normalized_vector(weights: dict[str, float]) -> dict[str, float]:
    norm = math.sqrt(sum(value * value for value in weights.values() if value > 0))
    if norm <= 0:
        return {}
    return {key: value / norm for key, value in weights.items() if value > 0}
