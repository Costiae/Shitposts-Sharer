"""群口味的更新逻辑：吃屎 → 口味漂移 → 可信度变化。

“吃屎偏好”就是群的关键词向量；可信度由两件事决定：
1. **样本量**：该群自己产生的、被判定为屎的消息条数（含字数的字面量，用来防“刷条数”）；
2. **偏移度**：新样本与既有口味差多远 —— 差得越远，说明这个群口味不稳，
   一次判定越不足以支撑下一次搬运。

两者都只做“温和惩罚”，因为需求说了“不必过于精确”。
"""

from __future__ import annotations

import math
from collections import defaultdict
from typing import Any, Mapping

from .config import Settings
from .models import GroupProfile, ShitRecord, normalized_vector
from .terms import DEFAULT_RULES, TermRules
from .vector import (
    blend_into,
    cosine,
    decay_weight,
    merge_profile_terms,
    merge_similar_terms,
)


def update_intake(
    group: GroupProfile,
    shit: ShitRecord,
    settings: Settings,
    *,
    is_native: bool = True,
) -> float:
    """把一个样本融进群口味，返回这次样本的“偏移度”（0~1）。

    ``is_native`` 表示这是群自己产的原生屎（能提高可信度），
    否则只是群收过的东西（只影响口味，不影响可信度）。
    """
    before = dict(group.intake)
    drift = 0.0
    if before and shit.vector:
        # 相似度越低 → 偏移越大
        drift = max(0.0, min(1.0, 1.0 - cosine(before, shit.vector)))

    weight = decay_weight(shit.ts)
    addend = {key: value * weight for key, value in shit.vector.items()}
    rules = settings.term_rules
    group.intake = blend_into(
        before, addend, momentum=0.92 if is_native else 0.97, rules=rules
    )
    # 每次新增样本都把整张口味表重新合并一遍：老词条（例如上次登记的
    # “抽象话”）和新词条（“抽象”）必须收敛到同一个概念，否则向量会被
    # 相近的写法稀释。别名/停用词/单字也在这一步清理。
    group.intake = merge_profile_terms(
        group.intake,
        threshold=settings.keyword_merge_threshold,
        rules=rules,
    )

    # 偏移度做指数平滑：单次跳题只让稳定度轻微下降，连续跳题才会明显掉。
    # 第一次样本（before 为空）本来就没有参照，drift 恒为 0，不予惩罚。
    if drift > 0 or group.drift > 0:
        momentum = 0.7 if is_native else 0.8
        group.drift = max(
            0.0, min(1.0, momentum * group.drift + (1.0 - momentum) * drift)
        )

    if is_native:
        group.native_message_count += 1
        group.native_char_count += max(1, len(shit.text_blob))
        group.last_shit_at = shit.ts
        # 采纳积分：偏移越大，这条屎对“这个大方向”的说服力越低
        credit = 1.0 - 0.5 * drift
        group.accepted_count += max(1, int(round(credit * 2)))

    return drift


def confidence_score(
    *,
    native_messages: int,
    native_chars: int,
    drift: float,
    accepted: int,
    rejected: int,
    k: float = 10.0,
) -> float:
    """可信度 = 样本饱和度 × 稳定度 × 采纳率，全部温和加权。"""
    # 1. 样本饱和度：用的是“有效原生屎”的条数与字数的几何平均
    eff_messages = max(0, native_messages)
    eff_chars = max(0, native_chars) / 40.0  # 40 字算一“份”有效样本
    effective = math.sqrt(max(0.0, eff_messages) * max(0.0, eff_chars)) or eff_messages
    saturation = effective / (effective + k) if (effective + k) > 0 else 0.0

    # 2. 稳定度：偏移越大越不稳定
    stability = 1.0 - 0.6 * max(0.0, min(1.0, drift))

    # 3. 采纳率：收下了多少 vs 被打回去多少
    total = max(0, accepted) + max(0, rejected)
    acceptance = 1.0 if total == 0 else max(0, accepted) / total

    return max(0.0, min(1.0, saturation * stability * (0.7 + 0.3 * acceptance)))


def apply_rejection(group: GroupProfile, settings: Settings) -> None:
    """目标群把屎打回来了（群友 T 掉了 / 判定不买账），降一点可信度。"""
    group.rejected_count += 1
    need = group.rejected_count * 3
    if group.accepted_count < need:
        # 证据不足以说明这个群真喜欢吃，减少可信的样本量
        group.native_message_count = max(0, group.native_message_count - 1)


def compact_intake(
    intake: dict[str, float],
    rules: TermRules = DEFAULT_RULES,
) -> dict[str, float]:
    """把已存的口味压成规范词：别名归一、丢停用词/单字、同根词合并。

    这一步会写回存储，主要用来清理旧版本留下的 ``站`` / ``玩梗`` / ``二次``
    这类碎片。合并后重新归一化，保持和正常更新一致的尺度。
    """
    if not intake:
        return {}
    return normalized_vector(merge_similar_terms(intake, rules))


def apply_term_rewrite(
    intake: dict[str, float],
    data: Mapping[str, Any],
    rules: TermRules = DEFAULT_RULES,
) -> dict[str, float]:
    """按 LLM 的清理结果重写口味向量。

    ``data`` 形如::

        {
          "drop": ["站", "玩梗"],
          "groups": [
            {"canonical": "人工智能", "members": ["ai", "人工智能"]}
          ]
        }

    只接受原词表里出现过的成员；模型胡编的未知词直接忽略，避免无中生有。
    ``canonical`` 可以不在原词表里（只要至少有一个成员在），这样模型能把
    ``ai`` 归到一个更规范的写法上。
    """
    if not intake:
        return {}

    known = {rules.canonical(term) for term in intake}
    drop_raw = data.get("drop") or []
    if isinstance(drop_raw, str):
        drop_raw = [drop_raw]
    elif not isinstance(drop_raw, (list, tuple, set)):
        drop_raw = []
    drop = {
        rules.canonical(str(term))
        for term in drop_raw
        if str(term).strip()
    }

    mapping: dict[str, str] = {}
    groups = data.get("groups") or []
    if isinstance(groups, list):
        for item in groups:
            if not isinstance(item, Mapping):
                continue
            canonical = rules.canonical(str(item.get("canonical") or ""))
            if not canonical:
                continue
            members = item.get("members") or []
            if isinstance(members, str):
                members = [members]
            elif not isinstance(members, (list, tuple, set)):
                continue
            for member in members:
                key = rules.canonical(str(member))
                if key in known:
                    mapping[key] = canonical
            if canonical in known:
                mapping[canonical] = canonical

    rewritten: dict[str, float] = defaultdict(float)
    for term, value in intake.items():
        key = rules.canonical(term)
        if key in drop or rules.is_stopped(key):
            continue
        rewritten[mapping.get(key, key)] += value
    return normalized_vector(merge_similar_terms(dict(rewritten), rules))


def intake_terms(
    group: GroupProfile,
    limit: int = 0,
    rules: TermRules = DEFAULT_RULES,
) -> list[tuple[str, float]]:
    """合并同根词后的口味词，按权重降序返回。

    存储层已经在 ``blend_into`` 里合并过；这里再合并一次是为了照顾
    旧版本留下的数据 —— 不重写库也能立刻得到干净的展示。
    """
    if not group.intake:
        return []
    merged = merge_similar_terms(group.intake, rules)
    items = sorted(merged.items(), key=lambda kv: kv[1], reverse=True)
    return items[:limit] if limit > 0 else items


def describe_intake(
    group: GroupProfile,
    limit: int = 10,
    rules: TermRules = DEFAULT_RULES,
) -> str:
    terms = intake_terms(group, limit, rules)
    if not terms:
        return "（还没有口味样本）"
    return "、".join(f"{key}({value:.2f})" for key, value in terms)
