"""匹配“屎 ↔ 群”的两种来源，以及混合判定。

需求原话：“做一个设置，两个都实现，开关，更改结果注入提示词。”

- 关键词（keyword）：零成本、可解释，先用它粗筛出候选群；
- LLM：把粗筛结果作为提示词注入，让模型判断“这条屎对不对这个群的胃口”。

两个开关都在配置里：``analysis_provider_id`` 为空 = 关闭 LLM 判定，
只剩关键词；关键词阈值调成 0 = 关键词几乎全放行，交给 LLM 收口。
"""

from __future__ import annotations

from dataclasses import dataclass

from .config import Settings
from .models import GroupProfile, RouteCandidate, ShitRecord
from .vector import explain_overlap, keyword_score


@dataclass(slots=True)
class KeywordMatcher:
    settings: Settings

    def score(self, shit: ShitRecord, group: GroupProfile) -> RouteCandidate:
        raw = keyword_score(
            shit.vector,
            group.intake,
            threshold=self.settings.keyword_threshold,
            tolerance=self.settings.keyword_tolerance,
        )
        overlap = explain_overlap(shit.vector, group.intake)
        reason = "共享话题：" + "、".join(overlap) if overlap else "没有明显共同关键词"
        return RouteCandidate(
            group_key=group.group_key,
            score=raw,
            confidence=group.confidence(self.settings.confidence_k),
            source="keyword",
            reason=reason,
        )

    def image_match(self, shit: ShitRecord, group: GroupProfile) -> tuple[float, str]:
        """只看图片视觉标签的匹配分。

        图片通常没有正文，文字关键词匹配会失灵；所以图片单独用
        ``image_vector`` 对群口味做一次余弦匹配，并走独立阈值。
        """
        if not shit.image_vector:
            return 0.0, "没有图片标签"
        raw = keyword_score(
            shit.image_vector,
            group.intake,
            threshold=self.settings.image_match_threshold,
            tolerance=self.settings.image_match_tolerance,
        )
        overlap = explain_overlap(shit.image_vector, group.intake)
        reason = "图片标签：" + "、".join(overlap) if overlap else "图片标签无交集"
        return raw, reason

    def rank(
        self, shit: ShitRecord, groups: list[GroupProfile], min_score: float = 0.0
    ) -> list[RouteCandidate]:
        candidates = []
        for group in groups:
            candidate = self.score(shit, group)
            if candidate.score > min_score:
                candidates.append(candidate)
        candidates.sort(key=lambda item: (item.final_score, item.score), reverse=True)
        return candidates


def merge_llm_verdict(
    keyword_candidates: list[RouteCandidate],
    llm_scores: dict[str, float],
) -> list[RouteCandidate]:
    """把 LLM 的评分合并回候选集，并统一改标记为 llm 来源。"""
    merged: list[RouteCandidate] = []
    for candidate in keyword_candidates:
        score = llm_scores.get(candidate.group_key)
        if score is None:
            continue
        merged.append(
            RouteCandidate(
                group_key=candidate.group_key,
                score=max(0.0, min(1.0, float(score))),
                confidence=candidate.confidence,
                source="llm",
                reason=candidate.reason,
            )
        )
    merged.sort(key=lambda item: (item.final_score, item.score), reverse=True)
    return merged


def take_top(candidates: list[RouteCandidate], limit: int) -> list[RouteCandidate]:
    if limit and limit > 0:
        return candidates[:limit]
    return candidates
