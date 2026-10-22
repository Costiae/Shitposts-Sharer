"""群口味的更新逻辑：吃屎 → 口味漂移 → 可信度变化。

“吃屎偏好”就是群的关键词向量；可信度由两件事决定：
1. **样本量**：该群自己产生的、被判定为屎的消息条数（含字数的字面量，用来防“刷条数”）；
2. **偏移度**：新样本与既有口味差多远 —— 差得越远，说明这个群口味不稳，
   一次判定越不足以支撑下一次搬运。

两者都只做“温和惩罚”，因为需求说了“不必过于精确”。
"""

from __future__ import annotations

import math

from .config import Settings
from .models import GroupProfile, ShitRecord
from .vector import blend_into, cosine, decay_weight


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
    group.intake = blend_into(before, addend, momentum=0.92 if is_native else 0.97)

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


def describe_intake(group: GroupProfile, limit: int = 10) -> str:
    if not group.intake:
        return "（还没有口味样本）"
    top = sorted(group.intake.items(), key=lambda kv: kv[1], reverse=True)[:limit]
    return "、".join(f"{key}({value:.2f})" for key, value in top)
