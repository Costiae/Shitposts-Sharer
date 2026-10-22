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
    *,
    merge_threshold: float = 0.6,
) -> dict[str, float]:
    """关键词加权 → 归一化向量。

    LLM 给出的关键词权重最高（它们是“话题脊”，不是噪音），
    正文分词次之，标签作为弱先验。强度只调整整体尺度，不改变方向。

    关键词会先做一次**相似合并**：``抽象 / 抽象话 / 抽象人`` 这类同一个
    概念的不同写法只按一个关键词计权，否则共享的双字窗口会被重复累加，
    让一个方向在向量里被人为放大。
    """
    weights: dict[str, float] = defaultdict(float)

    for keyword in merge_similar_keywords(keywords, threshold=merge_threshold):
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


# ---------------------------------------------------------------------- 关键词合并
_PUNCT_RE = re.compile(r"[\s\W_]+", re.UNICODE)


def _keyword_core(text: str) -> str:
    """去掉空格/标点并转小写，只留下比较用的“词核”。"""
    return _PUNCT_RE.sub("", str(text or "")).lower()


def _keyword_grams(text: str) -> set[str]:
    """字符二元组集合；与向量分词保持同一粒度。"""
    if not text:
        return set()
    if len(text) <= 2:
        return {text}
    return {text[index : index + 2] for index in range(len(text) - 1)}


def keyword_similarity(a: str, b: str) -> float:
    """两条关键词有多像（0~1）。

    两个信号取最大值：
    - 包含关系：短词是长词的子串，相似度 = 短/长（"抽象" ⊂ "抽象话" = 0.67）；
    - 字符二元组 Jaccard：处理顺序不同、只共享一部分的短语。
    """
    left, right = _keyword_core(a), _keyword_core(b)
    if not left or not right:
        return 0.0
    if left == right:
        return 1.0

    shorter, longer = sorted((left, right), key=len)
    containment = 0.0
    if len(shorter) >= 2 and shorter in longer:
        containment = len(shorter) / len(longer)

    left_grams, right_grams = _keyword_grams(left), _keyword_grams(right)
    union = left_grams | right_grams
    jaccard = len(left_grams & right_grams) / len(union) if union else 0.0
    return max(containment, jaccard)


def merge_similar_keywords(
    keywords: Iterable[str], threshold: float = 0.6
) -> list[str]:
    """把相似关键词合并成一条，保留先出现的顺序。

    合并后保留**更短**的写法（通常是更通用的那个），例如
    ``["抽象话", "抽象"]`` → ``["抽象"]``。阈值越低合并越激进。
    """
    merged: list[str] = []
    for raw in keywords:
        keyword = str(raw or "").strip()
        if not keyword:
            continue
        for index, existing in enumerate(merged):
            if keyword_similarity(keyword, existing) >= threshold:
                # 更短的写法更通用；长度相同则保留先出现的
                if len(keyword) < len(existing):
                    merged[index] = keyword
                break
        else:
            merged.append(keyword)
    return merged


def merge_profile_terms(
    profile: dict[str, float],
    threshold: float = 0.6,
    max_terms: int = 512,
) -> dict[str, float]:
    """把**整个口味词表**里的相似关键词合并，再重新归一化。

    与 ``merge_similar_keywords`` 的区别：那个只看一批新关键词，这个处理
    已经存在的老词条。群口味是长期累积的，之前登记过 ``抽象话``，今天又
    学到 ``抽象``，如果只在新增时去重，老词条会永远留在向量里并稀释
    权重。所以每次 ``update_intake`` 都要调用一次。

    合并时把相似词条的权重**相加**（它们本来就是同一个概念的证据），
    最后统一归一化，避免向量模长漂移。
    """
    if len(profile) < 2:
        return dict(profile)

    terms = [
        (str(term), float(weight))
        for term, weight in profile.items()
        if term and weight > 0
    ]
    if not terms:
        return {}

    # 并查集：相似关系要传递。只按权重顺序贪心聚类的话，
    # “抽象话 / 抽象人 / 抽象” 里如果先处理没有桥接的两条，就会漏合并。
    parent = list(range(len(terms)))

    def find(index: int) -> int:
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = parent[index]
        return index

    def union(left: int, right: int) -> None:
        root_left, root_right = find(left), find(right)
        if root_left != root_right:
            parent[root_right] = root_left

    for i in range(len(terms)):
        for j in range(i + 1, len(terms)):
            if keyword_similarity(terms[i][0], terms[j][0]) >= threshold:
                union(i, j)

    clusters: dict[int, list[tuple[str, float]]] = defaultdict(list)
    for index, item in enumerate(terms):
        clusters[find(index)].append(item)

    merged: dict[str, float] = {}
    for members in clusters.values():
        # 更短的写法更通用；长度相同则保留权重更高的
        canonical = min(members, key=lambda item: (len(item[0]), -item[1]))[0]
        merged[canonical] = sum(weight for _, weight in members)

    if len(merged) > max_terms:
        kept = sorted(merged.items(), key=lambda item: item[1], reverse=True)[:max_terms]
        merged = dict(kept)
    return normalized_vector(merged)


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
