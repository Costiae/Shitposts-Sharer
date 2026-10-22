"""搬运路由：决定一条屎去哪几个群。

规则（来自需求）：
1. 只有 emit / both 的群能当屎源；只有 receive / both 的群能收屎；
2. 可信度不够的群不搬 —— 但惩罚很温和，过了门槛就行；
3. 匹配分过阈值就搬，多个群都过就都搬；
4. 同一条屎对同一个群只搬一次（含“别人已经先搬过了”的情况）。
"""

from __future__ import annotations

from dataclasses import dataclass, field

from .config import Settings
from .models import Direction, GroupProfile, RouteCandidate, ShitRecord, normalize_text
from .similarity import KeywordMatcher


@dataclass(slots=True)
class RoutePlan:
    source_group: str
    candidates: list[RouteCandidate] = field(default_factory=list)
    rejected: list[tuple[str, str]] = field(default_factory=list)

    def group_keys(self) -> list[str]:
        return [item.group_key for item in self.candidates]


class Router:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.matcher = KeywordMatcher(settings)

    def eligible_groups(
        self, groups: list[GroupProfile], exclude: set[str] | None = None
    ) -> list[GroupProfile]:
        exclude = exclude or set()
        return [
            group
            for group in groups
            if group.enabled and group.direction.can_receive and group.group_key not in exclude
        ]

    def keyword_plan(
        self,
        shit: ShitRecord,
        groups: list[GroupProfile],
        exclude: set[str] | None = None,
        delivered: set[str] | None = None,
    ) -> RoutePlan:
        """只用关键词向量做一轮粗筛。"""
        plan = RoutePlan(source_group=shit.source_group)
        delivered = delivered or set()

        for group in self.eligible_groups(groups, exclude):
            if group.group_key in delivered:
                plan.rejected.append((group.group_key, "已经搬过"))
                continue

            candidate = self.matcher.score(shit, group)
            threshold = self.settings.threshold_for(group.threshold)

            # 图片专属门：文字话题对上了不代表图对胃口。图片必须靠视觉
            # 标签（角色/IP/属性）单独过门槛，达不到就不搬。
            if shit.image_vector:
                image_score, image_reason = self.matcher.image_match(shit, group)
                if image_score <= 0:
                    plan.rejected.append(
                        (
                            group.group_key,
                            f"{image_reason}（图片匹配 {image_score:.2f} < "
                            f"{self.settings.image_match_threshold:.2f}）",
                        )
                    )
                    continue
                candidate.image_score = image_score
                if image_score > candidate.score:
                    candidate.score = image_score

            # 两道独立的门：口味像不像（score）与这个群值得不值得信（confidence）
            if candidate.score <= 0:
                plan.rejected.append((group.group_key, candidate.reason))
                continue
            if candidate.confidence < threshold:
                plan.rejected.append(
                    (
                        group.group_key,
                        f"可信度不足 {candidate.confidence:.2f} < {threshold:.2f}",
                    )
                )
                continue

            plan.candidates.append(candidate)

        plan.candidates.sort(key=lambda item: (item.final_score, item.score), reverse=True)
        limit = int(self.settings.get("max_targets") or 0)
        if limit > 0:
            plan.candidates = plan.candidates[:limit]
        return plan

    def apply_llm_scores(
        self,
        plan: RoutePlan,
        llm_scores: dict[str, float],
        groups: list[GroupProfile],
    ) -> RoutePlan:
        """把 LLM 的契合度分数覆盖到关键词分数上，阈值判定保持不变。"""
        lookup = {group.group_key: group for group in groups}
        refined: list[RouteCandidate] = []

        for candidate in plan.candidates:
            score = llm_scores.get(candidate.group_key)
            if score is None:
                continue
            group = lookup.get(candidate.group_key)
            threshold = self.settings.threshold_for(group.threshold if group else 0.0)
            if score < threshold:
                plan.rejected.append(
                    (
                        candidate.group_key,
                        f"LLM 判定不对胃口 {score:.2f} < {threshold:.2f}",
                    )
                )
                continue
            refined.append(
                RouteCandidate(
                    group_key=candidate.group_key,
                    score=score,
                    confidence=candidate.confidence,
                    source="llm",
                    reason=candidate.reason,
                    image_score=candidate.image_score,
                )
            )

        refined.sort(key=lambda item: (item.final_score, item.score), reverse=True)
        plan.candidates = refined
        return plan

    def should_carry_shit(self, group: GroupProfile | None) -> bool:
        """这个群能不能把屎发出去。"""
        return bool(group and group.enabled and group.direction.can_emit)

    def can_receive(self, group: GroupProfile | None) -> bool:
        return bool(group and group.enabled and group.direction.can_receive)


def fingerprint_of(shit: ShitRecord) -> str:
    """内容指纹：图片按 URL + 标签，文本按规范化正文。

    图片类不能只用标签：同 tag 的两张不同图会被误判成重复；反过来，
    同一张图被不同压缩/转发时 URL 也会变 —— 两个信号都带上，能同时
    降低“误去重”和“漏去重”。
    """
    image_urls = [
        str(segment.get("url") or "")
        for segment in shit.payload
        if isinstance(segment, dict) and segment.get("kind") == "image"
    ]
    image_urls = [url for url in image_urls if url]

    text = normalize_text(shit.text_blob)
    if image_urls:
        tags = "/".join(sorted(shit.image_tags or shit.keywords))
        base = f"{shit.kind.value}|{'|'.join(image_urls)}|{tags}"
    elif len(text) >= 8:
        base = text
    else:
        base = f"{shit.kind.value}|{'/'.join(sorted(shit.keywords))}"
    return __import__("hashlib").sha1(base.encode("utf-8")).hexdigest()


def direction_matrix() -> dict[Direction, str]:
    """方向说明。主语永远是「这个群」。"""
    return {direction: direction.describe() for direction in Direction}
