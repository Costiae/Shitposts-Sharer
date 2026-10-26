"""口味词归一化：停用词 + 全称/缩写别名。

群口味只该记住“聊什么”，不该记住“这是一坨什么类型的屎”：
``玩梗``、``小时``、``时前`` 这类词没有区分度，统一丢进停用词。
同一个概念的多种写法（``人工智能`` / ``AI``）先归一到规范词，
再参与合并与计权，避免它们在口味表里各占一行。

两张表都允许在插件配置里追加，内置表只兜底最常见的噪音和缩写。
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any, Iterable, Mapping

# 无论正文还是 LLM 关键词，出现这些词都不进向量。
DEFAULT_STOPWORDS = frozenset(
    {
        # 平台/消息载体元数据：不是群喜欢的内容类型
        "b站", "哔哩哔哩", "bilibili", "qq", "用户", "图片", "图像",
        "视频", "链接", "分享", "app_type", "app", "卡片", "消息",
        # 时间戳 / 转发占位留下的滑窗碎片
        "小时", "时前", "分钟", "天前", "周前", "月前", "年前",
        "刚刚", "昨天", "今天", "明天", "前天", "后天",
        "上午", "下午", "晚上", "凌晨", "星期",
        "未展", "展开",
        # 只描述“这是一坨屎”，不描述话题
        "玩梗", "玩烂梗", "造梗", "接梗", "抛梗", "烂梗",
        "梗", "梗图",
    }
)

# 规范词 -> 常见别名（全称 / 缩写 / 俗称）。键值都会被 lower()。
DEFAULT_ALIASES: dict[str, tuple[str, ...]] = {
    "人工智能": ("ai", "artificial intelligence", "人工智慧"),
    "二次元": ("2次元",),
    "b站": ("哔哩哔哩", "bilibili"),
    "音乐游戏": ("音游",),
    "美少女游戏": ("galgame", "gal"),
}

_SPLIT = re.compile(r"[\s,，、;；]+")
_ALIAS_LINE = re.compile(r"^\s*([^=:：]+?)\s*[=:：]\s*(.+?)\s*$")


def parse_stopwords(raw: Any) -> set[str]:
    """把配置里的停用词解析成集合。支持逗号/空格/换行分隔，也接受列表。"""
    if raw is None:
        return set()
    if isinstance(raw, (list, tuple, set)):
        pieces: Iterable[Any] = raw
    else:
        pieces = _SPLIT.split(str(raw))
    return {str(piece).strip().lower() for piece in pieces if str(piece).strip()}


def parse_aliases(raw: Any) -> dict[str, tuple[str, ...]]:
    """解析别名配置。

    支持两种写法：
    - 每行 ``规范词=别名1,别名2``（WebUI 文本框推荐）
    - JSON 对象 ``{"规范词": ["别名1", "别名2"]}``
    """
    if raw is None or raw == "":
        return {}
    if isinstance(raw, Mapping):
        data: Any = raw
    else:
        text = str(raw).strip()
        try:
            data = json.loads(text) if text.startswith("{") else None
        except (TypeError, ValueError):
            data = None
        if data is None:
            data = {}
            for line in text.splitlines():
                match = _ALIAS_LINE.match(line)
                if not match:
                    continue
                canonical, aliases = match.groups()
                data[canonical] = [part for part in _SPLIT.split(aliases) if part]

    parsed: dict[str, tuple[str, ...]] = {}
    if not isinstance(data, Mapping):
        return parsed
    for canonical, aliases in data.items():
        name = str(canonical).strip().lower()
        if not name:
            continue
        if isinstance(aliases, str):
            values: Iterable[Any] = _SPLIT.split(aliases)
        else:
            values = aliases or ()
        parsed[name] = tuple(
            str(alias).strip().lower() for alias in values if str(alias).strip()
        )
    return parsed


@dataclass(frozen=True, slots=True)
class TermRules:
    """停用词与别名索引的只读组合。"""

    stopwords: frozenset[str]
    aliases: dict[str, str]  # 任意写法 -> 规范词

    def canonical(self, term: str) -> str:
        lowered = (term or "").strip().lower()
        return self.aliases.get(lowered, lowered)

    def is_stopped(self, term: str) -> bool:
        return not term or self.canonical(term) in self.stopwords

    @classmethod
    def build(cls, stopwords_raw: Any = None, aliases_raw: Any = None) -> "TermRules":
        stopwords = set(DEFAULT_STOPWORDS) | parse_stopwords(stopwords_raw)

        index: dict[str, str] = {}

        def register(canonical: str, aliases: Iterable[str]) -> None:
            name = canonical.strip().lower()
            if not name:
                return
            index[name] = name
            for alias in aliases:
                alias = str(alias).strip().lower()
                if alias:
                    index[alias] = name

        for canonical, aliases in DEFAULT_ALIASES.items():
            register(canonical, aliases)
        # 用户配置放最后，可以覆盖内置映射
        for canonical, aliases in parse_aliases(aliases_raw).items():
            register(canonical, aliases)

        return cls(frozenset(stopwords), index)


DEFAULT_RULES = TermRules.build()
