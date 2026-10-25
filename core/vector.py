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
from .terms import DEFAULT_RULES, TermRules

# 中英文分词：连续汉字按 2 字滑窗，英文/数字按单词
_CJK = re.compile(r"[\u4e00-\u9fff]+")
_WORD = re.compile(r"[A-Za-z][A-Za-z0-9_]{1,}|[0-9]{2,}")
# 关键词/标签内部的分隔符：模型偶尔会把多个词塞进同一个字符串
_TERM_SEPARATORS = re.compile(r"[\s,，、;；:：/|·\-—~～]+")
# 词条两端的标点/引号
_TERM_EDGE = re.compile(r"^[^\w\u4e00-\u9fff]+|[^\w\u4e00-\u9fff]+$")

_STOP_CHARS = set("的了是我你他她它们在有不一个人这那里就都而及与或等啊吧呢嘛哦啊嗯诶呀")
_STOP_WORDS = {
    "the", "and", "for", "you", "are", "this", "that", "with", "have",
    "http", "https", "www", "com", "cn", "bilibili", "b23", "bv",
}

# 单条 token 最长长度。超长串通常是邀请码 / 短链 / 哈希，不是话题。
_MAX_TOKEN_CHARS = 32
_MAX_DIGIT_TOKEN = 8
# 20 位以上、同时含字母和数字的 ASCII 串，大概率是邀请码/标识符。
_INVITE_RE = re.compile(r"^(?=.*[A-Za-z])(?=.*\d)[A-Za-z0-9_]{20,}$")
# 汉字与多段数字交错（如“鸡8狗10”）：典型的分词/截取乱码，不是话题。
_CJK_DIGIT_NOISE_RE = re.compile(r"^(?:[\u4e00-\u9fff]+\d+){2,}$")


def _is_noise_token(token: str) -> bool:
    """过滤超长串、邀请码/哈希、长数字 ID 和 CJK/数字交错乱码。"""
    if not token:
        return True
    if len(token) > _MAX_TOKEN_CHARS:
        return True
    if _INVITE_RE.match(token):
        return True
    if _CJK_DIGIT_NOISE_RE.match(token):
        return True
    if token.isdigit() and len(token) > _MAX_DIGIT_TOKEN:
        return True
    return False


# 2 字滑窗不认词边界，正文里必然切出“小时/时前”这类碎片。
# 真正的词组由 keywords/tags 整词进入向量；正文里用 TermsRules 的
# 停用词表拦掉这些碎片（见 core/terms.py）。

# 中英混排（“B站转发”“3小时前”）在切块前先断开，避免跨脚本切出“站转”。
_SCRIPT_EDGE = re.compile(
    r"(?<=[A-Za-z0-9])(?=[\u4e00-\u9fff])|(?<=[\u4e00-\u9fff])(?=[A-Za-z0-9])"
)


DEFAULT_HALF_LIFE_DAYS = 30.0


def tokenize(
    text: str,
    grams: int = 2,
    exclude: Iterable[str] = (),
    rules: TermRules = DEFAULT_RULES,
) -> list[str]:
    """把正文切成候选关键词（只是低权重的召回信号）。

    注意：这里只服务正文。LLM 给出的 keywords/tags 本身已经是词组，
    走 :func:`_term_tokens` 整词入向量，不会再被滑窗切开；单个汉字
    歧义太大（“B站”只会剩下个“站”），正文里直接丢掉。

    ``exclude`` 里的整词在正文里出现时，跨越它边界的滑窗也一并跳过：
    “抽象离谱”不该再切出“象离”，“二次元机器人”不该切出“元机”。
    ``rules`` 负责停用词与别名归一（``AI`` → ``人工智能``）。
    """
    text = _SCRIPT_EDGE.sub(" ", text or "")
    terms = [str(term) for term in exclude if term]
    # 停用词也按“整词占位”处理：拦掉“玩梗”本身，也拦掉跨过它的“梗内”。
    blocked = [*terms, *rules.stopwords]
    tokens: list[str] = []
    for chunk in _CJK.findall(text):
        if len(chunk) < grams:
            continue
        covered = _covered_positions(chunk, blocked)
        for i in range(len(chunk) - grams + 1):
            if any(covered[i : i + grams]):
                continue
            gram = rules.canonical(chunk[i : i + grams])
            if rules.is_stopped(gram):
                continue
            if any(ch in _STOP_CHARS for ch in gram):
                continue
            tokens.append(gram)
    for word in _WORD.findall(text.lower()):
        if len(word) < 2:
            continue
        word = rules.canonical(word)
        if word in _STOP_WORDS or rules.is_stopped(word):
            continue
        if _is_noise_token(word):
            continue
        tokens.append(word)
    return tokens


def _covered_positions(chunk: str, terms: Iterable[str]) -> list[bool]:
    """标出正文里已经由整词关键词覆盖的字符位置。"""
    covered = [False] * len(chunk)
    for term in terms:
        if not term or not _CJK.fullmatch(term):
            continue
        start = chunk.find(term)
        while start >= 0:
            for index in range(start, start + len(term)):
                covered[index] = True
            start = chunk.find(term, start + 1)
    return covered


def _term_tokens(term: str, rules: TermRules = DEFAULT_RULES) -> list[str]:
    """把一个关键词/标签拆成“整词”，而不是 2 字滑窗碎片。

    ``二次元`` → ``["二次元"]``，不会变成 ``二次``/``次元``。
    模型偶尔会在一个字段里用空格或斜杠并列多个词，这里一并拆开。
    最后再走一遍停用词与别名归一（``AI`` → ``人工智能``）。
    """
    tokens: list[str] = []
    for piece in _TERM_SEPARATORS.split(str(term or "")):
        piece = _TERM_EDGE.sub("", piece.strip().lower())
        if not piece:
            continue
        piece = rules.canonical(piece)
        if piece in _STOP_WORDS or rules.is_stopped(piece):
            continue
        if _is_noise_token(piece):
            continue
        if len(piece) == 1:
            if piece in _STOP_CHARS:
                continue
        elif all(ch in _STOP_CHARS for ch in piece):
            continue
        tokens.append(piece)
    return tokens


def tokenize_keyword(term: str, rules: TermRules = DEFAULT_RULES) -> list[str]:
    """公开的关键词分词入口：关键词/标签按“整词”进入向量。

    与正文的 2 字滑窗不同，``二次元`` 仍然是 ``二次元``，不会碎成
    ``二次``/``次元``；混合词 ``b站``、``鸡8狗10`` 也会完整保留。
    这个函数主要给测试和调试使用，正常构建走 :func:`build_vector`。
    """
    return _term_tokens(term, rules)


def _normalize_terms(
    weights: dict[str, float],
    rules: TermRules,
) -> dict[str, float]:
    """别名归一、停用词过滤、丢单字，并把同一规范词的权重相加。"""
    normalized: dict[str, float] = defaultdict(float)
    for term, value in weights.items():
        canonical = rules.canonical(str(term))
        if canonical in _STOP_WORDS or rules.is_stopped(canonical):
            continue
        if len(canonical) < 2:
            continue
        if _is_noise_token(canonical):
            continue
        try:
            numeric = float(value)
        except (TypeError, ValueError):
            continue
        if numeric <= 0:
            continue
        normalized[canonical] += numeric
    return dict(normalized)


def merge_similar_terms(
    weights: dict[str, float],
    rules: TermRules = DEFAULT_RULES,
) -> dict[str, float]:
    """先归一别名、丢掉停用词，再合并同根词。

    ``机器`` / ``机器人``、``二次元`` / ``二次元文化`` 会并成一项，
    权重相加；代表词取当前权重最高（同权取更长）的那个。
    ``AI`` 与 ``人工智能`` 靠 ``rules`` 的别名表归到同一个词。

    单字词一律丢掉：它们几乎不含话题信息，多半是旧版滑窗切出来的碎片
    （“站”来自“B站”），留着只会在展示里占位。真正有意义的单字黑话
    交给 LLM 清理，或者写进别名表归到更长的规范词上。

    只处理“包含关系”和显式别名。``代码`` 与 ``编程`` 这类没有词表的
    同义词不在这里硬猜（可写进插件的别名配置，或用 ``/shit tidy``）。
    """
    weights = _normalize_terms(weights, rules)
    if len(weights) < 2:
        return weights

    terms = sorted(weights, key=lambda term: (len(term), term))
    parent = {term: term for term in terms}

    def find(term: str) -> str:
        root = term
        while parent[root] != root:
            root = parent[root]
        while parent[term] != root:
            parent[term], term = root, parent[term]
        return root

    for index, short in enumerate(terms):
        for later in range(index + 1, len(terms)):
            longer = terms[later]
            if short not in longer:
                continue
            left, right = find(short), find(longer)
            if left != right:
                parent[right] = left

    groups: dict[str, list[str]] = defaultdict(list)
    for term in terms:
        groups[find(term)].append(term)

    merged: dict[str, float] = {}
    for members in groups.values():
        # 同权时取更长的一个：旧数据里的短词多半是被截断的碎片。
        representative = max(members, key=lambda term: (weights[term], len(term)))
        merged[representative] = sum(weights[term] for term in members)
    return merged


def normalize_intake_terms(
    profile: dict[str, float],
    max_terms: int = 512,
    rules: TermRules = DEFAULT_RULES,
) -> dict[str, float]:
    """把旧版/外部写入的口味表压成当前整词空间。

    旧数据可能是 2 字滑窗碎片（``二次``/``次元``/``站``）、停用词
    （``玩梗``）或未归一的别名；这里用 :func:`merge_similar_terms` 统一
    清理：丢单字/停用词、别名归一、同根词合并，并优先保留权重最高
    （同权更长）的代表词。对已经是现代整词的口味表是幂等的，可在迁移时
    安全重复执行。
    """
    if not profile:
        return {}
    merged = merge_similar_terms(profile, rules)
    if len(merged) > max_terms:
        kept = sorted(merged.items(), key=lambda item: item[1], reverse=True)[
            :max_terms
        ]
        merged = dict(kept)
    return normalized_vector(merged)


def build_vector(
    keywords: Iterable[str],
    text: str,
    intensity: float = 0.0,
    tags: Iterable[str] = (),
    *,
    merge_threshold: float = 0.6,
    rules: TermRules = DEFAULT_RULES,
) -> dict[str, float]:
    """关键词加权 → 归一化向量。

    LLM 给出的关键词权重最高（它们是“话题脊”，不是噪音），
    正文分词次之，标签作为弱先验。强度只调整整体尺度，不改变方向。

    关键词会先做一次**相似合并**：``抽象 / 抽象话 / 抽象人`` 这类同一个
    概念的不同写法只按一个关键词计权，否则会在向量里被人为放大。
    合并后按**整词**计权：``二次元`` 就是 ``二次元``，不会再被滑窗拆成
    ``二次``/``次元`` —— 否则群口味会变成一堆碎片。
    """
    weights: dict[str, float] = defaultdict(float)

    # 整词表用于压制正文里的同源碎片（正文里出现“二次元”时不要再加“二次”）。
    atomic_terms: list[str] = []
    for keyword in merge_similar_keywords(keywords, threshold=merge_threshold):
        for term in _term_tokens(keyword, rules):
            weights[term] += 3.0
            if term not in atomic_terms:
                atomic_terms.append(term)

    for tag in tags:
        for term in _term_tokens(tag, rules):
            weights[term] += 0.5
            if term not in atomic_terms:
                atomic_terms.append(term)

    for token in tokenize(text, exclude=atomic_terms, rules=rules):
        if any(
            token == term or (len(term) > len(token) and token in term)
            for term in atomic_terms
        ):
            continue
        weights[token] += 1.0

    weights = merge_similar_terms(weights, rules)
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
    rules: TermRules = DEFAULT_RULES,
) -> dict[str, float]:
    """把**整个口味词表**里的相似关键词合并，再重新归一化。

    合并前先走一遍 ``rules``：别名归一、停用词、丢单字。旧版本留下的
    ``站`` / ``玩梗`` / ``AI`` 会在这里被清掉或并到规范词上。

    与 ``merge_similar_keywords`` 的区别：那个只看一批新关键词，这个处理
    已经存在的老词条。群口味是长期累积的，之前登记过 ``抽象话``，今天又
    学到 ``抽象``，如果只在新增时去重，老词条会永远留在向量里并稀释
    权重。所以每次 ``update_intake`` 都要调用一次。

    合并时把相似词条的权重**相加**（它们本来就是同一个概念的证据），
    最后统一归一化，避免向量模长漂移。
    """
    profile = _normalize_terms(profile, rules)
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
    rules: TermRules = DEFAULT_RULES,
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

    # 别名归一 + 停用词过滤 + 同根词合并：
    # 老版本留下的“二次”“次元”“玩梗”会在这一步并掉 / 丢掉。
    merged = merge_similar_terms(merged, rules)

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


def adaptive_keyword_threshold(
    group_vector: dict[str, float],
    threshold: float,
    tolerance: float,
) -> tuple[float, float]:
    """按群口味的集中度收紧/放宽关键词门槛。

    群口味是多个话题的归一化混合向量：如果最高权重只有 0.5，那么
    新屎即使命中这个最强话题，余弦也很难达到 0.75。此时还要求配置值
    是数学上不可能的，会导致候选在关键词层就被淘汰，LLM 复核永远跑不到。

    这里把配置值当作**上限**：实际门槛取
    ``min(配置值, max(0.30, 最强 token × 0.75))``，给“一条新屎通常有
    多个 token”留出余量；同时按实际门槛缩放容差，避免为了救回多话题群
    把门槛放得过松。
    """
    top = max((value for value in group_vector.values() if value > 0), default=0.0)
    if top <= 0:
        return threshold, tolerance
    effective = min(threshold, max(0.30, top * 0.75))
    effective_tolerance = min(tolerance, max(0.02, effective * 0.20))
    return effective, effective_tolerance


def keyword_score(
    shit_vector: dict[str, float],
    group_vector: dict[str, float],
    threshold: float = 0.75,
    tolerance: float = 0.10,
    *,
    adaptive: bool = False,
) -> float:
    """关键词来源的打分：相似度过了阈值就算合格，靠近阈值时线性收敛。

    这样“差不多”也能过——需求就是“不必过于精确”，
    但远离阈值的群不会因为算了个小数就被搬。

    ``adaptive=True`` 时会在比较前按群口味集中度调整门槛，专门处理
    “长期累积、话题很多”的群；直接调用本函数做单元测试时默认关闭，
    保持原有阈值语义。

    容差不取 0.05 这种刀尖值：中文双字滑窗的会话语料很稀疏，
    同话题的两段话余弦常常只有 0.6~0.8，卡太死会一条都搬不出去。
    """
    if adaptive:
        threshold, tolerance = adaptive_keyword_threshold(
            group_vector, threshold, tolerance
        )
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
