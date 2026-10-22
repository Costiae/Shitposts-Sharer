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
    """**群**在搬运拓扑中的角色（不是插件的视角）。

    RECEIVE  这个群收别人搬来的屎
    EMIT     这个群产屎，被别人搬走
    BOTH     两者都是

    字段值与 ``can_receive`` / ``can_emit`` 对齐，避免 in/out 那样的主语歧义。
    旧的 ``in`` / ``out`` 值由 storage 的迁移在加载时自动转换。
    """

    RECEIVE = "receive"
    EMIT = "emit"
    BOTH = "both"

    @property
    def can_receive(self) -> bool:
        return self in (Direction.RECEIVE, Direction.BOTH)

    @property
    def can_emit(self) -> bool:
        return self in (Direction.EMIT, Direction.BOTH)

    @classmethod
    def parse(cls, raw: str) -> "Direction":
        value = (raw or "").strip().lower()
        # 只认英文；语言统一，不再保留中文别名
        alias = {
            "receive": cls.RECEIVE,
            "recv": cls.RECEIVE,
            "emit": cls.EMIT,
            "both": cls.BOTH,
            "all": cls.BOTH,
        }
        if value in alias:
            return alias[value]
        for member in cls:
            if member.value == value:
                return member
        raise ValueError(
            f"unknown direction: {raw!r} (use receive / emit / both)"
        )

    def label(self) -> str:
        return {
            self.RECEIVE: "receive",
            self.EMIT: "emit",
            self.BOTH: "both",
        }[self]

    def describe(self) -> str:
        """给人看的一句话说明，主语永远是「这个群」。"""
        return {
            self.RECEIVE: "this group receives shit from others; it is never used as a source",
            self.EMIT: "this group produces shit that gets relayed out; it receives nothing",
            self.BOTH: "this group both produces shit and receives shit from others",
        }[self]


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
        if self.kind == "forward":
            # 未展开的合并转发：明确说“看不到内容”，而不是伪装成媒体
            return pad + "[合并转发：内容未展开]"
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
    # 口味偏移度的平滑值（0~1）：越大说明这个群最近的口味跳得越厉害。
    # 由 profile.update_intake 维护，参与可信度的“稳定度”因子。
    drift: float = 0.0
    native_message_count: int = 0
    native_char_count: int = 0
    accepted_count: int = 0
    rejected_count: int = 0
    last_shit_at: float = 0.0
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)

    # ---- 可信度 ----
    def confidence(self, k: float = 10.0) -> float:
        """可信度：与 ``profile.confidence_score`` 完全同一套公式。

        这里必须和指令层展示、路由层判定共用同一个实现，否则会出现
        “/shit status 说够格、实际却不搬”这种自相矛盾。

        延迟 import 是为了避免 models ↔ profile 的循环依赖：profile 在
        模块顶层 import models，而本方法只在真正计算时才需要它。
        """
        from .profile import confidence_score

        return confidence_score(
            native_messages=self.native_message_count,
            native_chars=self.native_char_count,
            drift=self.drift,
            accepted=self.accepted_count,
            rejected=self.rejected_count,
            k=k,
        )


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
    # 图片专属：视觉标签（角色 / IP / 属性 / 风格）与独立向量。
    # 文本屎这两个字段为空，走原来的关键词匹配即可。
    image_tags: list[str] = field(default_factory=list)
    image_vector: dict[str, float] = field(default_factory=dict)
    nsfw: bool = False
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
    # ---- 图片标记（无图时全为空） ----
    image_tags: list[str] = field(default_factory=list)
    characters: list[str] = field(default_factory=list)
    copyrights: list[str] = field(default_factory=list)
    nsfw: bool = False
    # 安全闸门：涉及未成年人的性化内容，永不搬运
    minor_sexual: bool = False


@dataclass(slots=True)
class RouteCandidate:
    """一个目标群的匹配结果。"""

    group_key: str
    score: float
    confidence: float
    source: str = "keyword"  # keyword | llm
    reason: str = ""
    # 图片专属：只看图片标签时的匹配分（0 表示没有图片标签或未匹配）
    image_score: float = 0.0

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
