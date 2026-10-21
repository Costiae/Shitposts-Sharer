"""关键词向量：构建、持久化、相似度。

“不必过于精确”——所以这里刻意不用 embedding 模型，也不跑分词器，
而是把关键词抽出来、各自加权、归一化成稀疏向量，再做余弦相似度。
好处是可解释（能直接说“因为你们都有 X、Y”）、零额外依赖、零额外费用。
"""

from __future__ import annotations

import math
import re
import time
from collections import defaultdict
from typing import Iterable

from .models import normalized_vector

# 中英文分词：连续汉字按 2 字滑窗，英文/数字按单词
_CJK = re.compile(r"[\u4e00-\u9fff]+")
_WORD = re.compile(r"[A-Za-z][A-Za-z0-9_]{1,}|[0-9]{2,}")

_STOP_CHARS = set("的了是我你他她它们在有不一个人这那里就都而及与或等啊吧呢嘛哦啊嗯诶呀")
_STOP_WORDS = {
    "the", "and", "for", "you", "are", "this", "that", "with", "have",
    "http", "https", "www", "com", "cn", "bilibili", "b23", "bv",
}

DEFAULT_HALF_LIFE_DAYS = 30.0


def tokenize(text: str, grams: int = 2) -> list[str]:
    """把文本切成候选关键词。"""
    text = text or ""
    tokens: list[str] = []
    for chunk in _CJK.findall(text):
        if len(chunk) == 1:
            if chunk not in _STOP_CHARS:
                tokens.append(chunk)
            continue
        for i in range(len(chunk) - grams + 1):
            gram = chunk[i : i + grams]
            if any(ch in _STOP_CHARS for ch in gram):
                continue
            tokens.append(gram)
    for word in _WORD.findall(text.lower()):
        if word in _STOP_WORDS or len(word) < 2:
            continue
        tokens.append(word)
    return tokens


def build_vector(
    keywords: Iterable[str],
    text: str,
    intensity: float = 0.0,
    tags: Iterable[str] = (),
) -> dict[str, float]:
    """关键词加权 → 归一化向量。

    LLM 给出的关键词权重最高（它们是“话题脊”，不是噪音），
    正文分词次之，标签作为弱先验。强度只调整整体尺度，不改变方向。
    """
    weights: dict[str, float] = defaultdict(float)

    for keyword in keywords:
        for token in tokenize(str(keyword)):
            weights[token] += 3.0

    for token in tokenize(text):
        weights[token] += 1.0

    for tag in tags:
        for token in tokenize(str(tag)):
            weights[token] += 0.5

    if not weights:
        return {}

    scale = 1.0 + max(0.0, min(intensity, 1.0))
    return normalized_vector({key: value * scale for key, value in weights.items()})


def blend_into(
    profile: dict[str, float],
    addend: dict[str, float],
    momentum: float = 0.92,
    max_terms: int = 512,
) -> dict[str, float]:
    """把新样本融入群向量，带惯性衰减，避免一次异常内容带歪整个口味。

    群口味不是精确的统计量，更像“氛围”，所以要钝、要滞后。
    """
    merged: dict[str, float] = {}
    for key, value in profile.items():
        merged[key] = value * momentum
    for key, value in addend.items():
        merged[key] = merged.get(key, 0.0) + value
    if not merged:
        return {}

    # 保底：只保留权重最高的若干项，防止关键词表无限膨胀
    if len(merged) > max_terms:
        top = sorted(merged.items(), key=lambda item: item[1], reverse=True)[:max_terms]
        merged = dict(top)

    return normalized_vector(merged)


def cosine(a: dict[str, float], b: dict[str, float]) -> float:
    """归一化向量的余弦相似度，自己实现以免引 numpy。"""
    if not a or not b:
        return 0.0
    if len(a) > len(b):
        a, b = b, a
    dot = 0.0
    for key, value in a.items():
        other = b.get(key)
        if other:
            dot += value * other
    return max(0.0, min(1.0, dot))


def explain_overlap(
    a: dict[str, float], b: dict[str, float], limit: int = 5
) -> list[str]:
    """列出两个向量共享的高权重词，用于告诉群友“为什么搬给你”。"""
    shared: list[tuple[str, float]] = []
    for key, value in a.items():
        other = b.get(key)
        if other:
            shared.append((key, value * other))
    shared.sort(key=lambda item: item[1], reverse=True)
    return [key for key, _ in shared[:limit]]


def decay_weight(ts: float, half_life_days: float = DEFAULT_HALF_LIFE_DAYS) -> float:
    """时间衰减系数：越老的屎对群口味的贡献越小。"""
    if half_life_days <= 0:
        return 1.0
    age_days = max(0.0, (time.time() - ts) / 86400.0)
    return 0.5 ** (age_days / half_life_days)


def keyword_score(
    shit_vector: dict[str, float],
    group_vector: dict[str, float],
    threshold: float = 0.75,
    tolerance: float = 0.10,
) -> float:
    """关键词来源的打分：相似度过了阈值就算合格，靠近阈值时线性收敛。

    这样“差不多”也能过——需求就是“不必过于精确”，
    但远离阈值的群不会因为算了个小数就被搬。

    容差不取 0.05 这种刀尖值：中文双字滑窗的会话语料很稀疏，
    同话题的两段话余弦常常只有 0.6~0.8，卡太死会一条都搬不出去。
    """
    similarity = cosine(shit_vector, group_vector)
    if similarity >= threshold:
        return similarity
    lower = max(0.0, threshold - tolerance)
    if similarity <= lower:
        return 0.0
    # 落在 [lower, threshold) 之间：给一个从 0 连续升到 threshold 的分
    span = max(threshold - lower, 1e-9)
    return similarity * (similarity - lower) / span


def mean_vector(vectors: Iterable[dict[str, float]]) -> dict[str, float]:
    total: dict[str, float] = defaultdict(float)
    count = 0
    for vector in vectors:
        for key, value in vector.items():
            total[key] += value
        count += 1
    if count == 0:
        return {}
    return normalized_vector({key: value / count for key, value in total.items()})


def vector_norm(vector: dict[str, float]) -> float:
    return math.sqrt(sum(value * value for value in vector.values() if value > 0))
