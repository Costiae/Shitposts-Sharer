"""LLM 分析器：判断“是不是屎”、它与上下文的关系、以及它对每个候选群的口味契合度。

三个提示词，全部要求模型输出 JSON —— 因为后面要拿关键词去算向量、拿分数去路由。
“不必过于精确”——但结构必须稳定，否则整条链路会散。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .accumulator import Suspicion
from .bili import BiliInfo
from .config import Settings
from .detector import summarise_segments
from .llm import LLMClient
from .models import Analysis, GroupProfile

SYSTEM = (
    "你是一个群聊搬运工的鉴屎员。你的任务是从一坨群聊内容里判断它是否是"
    "值得跨群传播的“屎”——即离谱、抽象、猎奇、好笑、或强烈符合某个群口味的"
    "非日常内容；正常闲聊、日常问候、普通新闻、工作讨论不是屎。"
    "打分时请参照给定的锚点，不要让分数集中在中低段——"
    "如果内容确实值得搬，就大胆给高分。"
    "图片类内容要按“角色、作品/IP、属性、风格”打结构化标签，供口味匹配使用；"
    "不要输出露骨描写，只输出标签和结构化字段。"
    "如果图片涉及未成年人的性化内容，必须设置 minor_sexual=true 且 is_shit=false，"
    "这类内容永远不允许跨群搬运。"
    "你必须只输出一个 JSON 对象，不要输出任何解释性文字。"
)

# 锚点很重要：只写“0.0 到 1.0 的离谱程度”时，模型没有校准参照，
# 会把“有点意思”的东西打在 0.2~0.5，导致 intensity 永远压在阈值下面。
INTENSITY_ANCHORS = """  0.0-0.2  完全正常，日常信息、问好、语气词
  0.3-0.4  有点意思但很常见（普通玩梗、轻微吐槽）
  0.5-0.6  明显抽象，值得跨群分享
  0.7-0.8  非常离谱，适合主动搬运
  0.9-1.0  逆天级，看到就想立刻转发"""

DEFAULT_SHIT_PROMPT = """请判断下面这坨群聊内容是不是“屎”（是否值得跨群搬运）。

来源群：{group_name}
载体：{kind}
粗筛触发：{trigger}
内容：
---
{content}
---

{BILI_BLOCK}
{CONTEXT_BLOCK}
判断标准（宽松，宁可误判也别漏掉有意思的）：
- 离谱/抽象的言论、发言或故事
- 猎奇、逆天、反直觉的内容
- 强烈玩梗、黑话、圈内梗
- 好笑且适合传播的图片或视频分享
- 二次元/成人向的图片（露点、强烈性暗示、擦边）如果足够好看、稀有或对味，也算屎；
  不要仅因为露骨就判否
- 图片必须打结构化视觉标签：角色名、作品/IP、属性（发色/服装/姿势/场景）、画风
- 正常聊天、问好、普通表情、日常信息 都不是屎

注意：请同时判断上面候选内容与上下文之间有没有“话题关系”——
也就是说，这段内容是被群友顺着什么话题引出来的，还是孤立地突然砸进来的。

intensity 请按下面的锚点打分，不要给保守值：
{anchors}

注意：is_shit 与 intensity 是两个独立判断。
is_shit=false 时 intensity 可以为 0；但只要你判了 is_shit=true，
就说明它确实值得传播，此时 intensity 应当落在 0.5 以上。

安全要求：如果图片涉及未成年人的性化内容，必须 minor_sexual=true 且 is_shit=false。
不要输出任何露骨描写。

只输出 JSON：
{{
  "is_shit": true/false,
  "intensity": 0.0 到 1.0，按上面的锚点打分,
  "tags": ["离谱", "玩梗", ...] 最多 5 个,
  "keywords": ["最能代表这坨屎话题的 3-8 个关键词，中文短语优先"],
  "image_tags": ["图片视觉标签：角色/作品/IP/属性/风格，最多 8 个；无图填空数组"],
  "characters": ["图中角色名，最多 5 个；无图填空数组"],
  "copyrights": ["作品/IP 名，最多 3 个；无图填空数组"],
  "nsfw": true/false,  // 是否成人向
  "minor_sexual": true/false,  // 是否涉及未成年人的性化内容（安全闸门）
  "reason": "一句话理由，20 字以内",
  "context_relevant": true/false,  // 是否与上文话题相关
  "context_keywords": ["从上下文中提炼的 2-5 个关键词"],
  "topic_seed": "如果它与上下文相关，用一句话概括这整个话题；否则填空字符串"
}}"""

DEFAULT_CONTEXT_PROMPT = """下面是群聊里的一段对话。请判断这些消息之间有没有上下文/话题关系。

内容：
---
{messages}
---

只输出 JSON：
{{
  "same_topic": true/false,
  "topics": ["如果有共同话题，列出 1-3 个"],
  "keywords": ["最能概括这段对话的 3-8 个关键词"],
  "summary": "一句话概括这段对话在聊什么",
  "intensity": 0.0 到 1.0，整段对话有多“屎”（离谱程度），锚点：
{anchors}
}}"""

DEFAULT_SAFETY_PROMPT = """只做安全判断，不要判断内容是否有趣。

请看下面的图片，判断它是否涉及未成年人的性化内容（包括明显未成年角色、
儿童外貌、校服/幼态化与性暗示的组合等）。

只输出 JSON：
{
  "minor_sexual": true/false
}

如果图片与未成年人无关，输出 false。如果无法看清图片，输出 null；
不要猜测。不要输出解释。"""

DEFAULT_GROUP_PROMPT = """你在给一个群挑它会不会喜欢的屎。

目标群：{group_name}
这个群的口味标签：{intake}
这个群最近吃过的屎关键词：{recent_keywords}

待判定的屎：
---
{shit_text}
---
关键词：{shit_keywords}

请判断这条屎有多符合这个群的口味，给 0.0 到 1.0 的分数。
0.0 = 完全不对胃口，1.0 = 简直为他们量身定做。

只输出 JSON：
{{
  "scores": {{"群标识": 0.0}},
  "reason": "一句话理由"
}}"""


@dataclass(slots=True)
class ShitVerdict:
    analysis: Analysis
    raw: dict[str, Any]
    ok: bool = True


class Analyzer:
    def __init__(self, settings: Settings, llm: LLMClient) -> None:
        self.settings = settings
        self.llm = llm

    def _prompt(self, key: str, default: str) -> str:
        """允许用配置覆盖内置提示词（按 Python format 模板，缺的占位符自动补空）。"""
        custom = str(self.settings.get(key) or "").strip()
        return custom or default

    def _secure_format(self, template: str, values: dict[str, str]) -> str:
        """先替换已知占位符，再把剩下的花括号模板交给 format 处理。

        用户自定义提示词里写漏了占位符（例如只写 {content}）也不该报错，
        所以这里做一层容错。
        """
        merged = dict(values)
        merged.setdefault("group_name", "")
        merged.setdefault("kind", "")
        merged.setdefault("trigger", "")
        merged.setdefault("content", "")
        merged.setdefault("BILI_BLOCK", "")
        merged.setdefault("CONTEXT_BLOCK", "")
        merged.setdefault("messages", "")
        merged.setdefault("anchors", INTENSITY_ANCHORS)  # 自定义提示词也要能用
        merged.setdefault("shit_text", "")
        merged.setdefault("shit_keywords", "")
        merged.setdefault("intake", "")
        merged.setdefault("recent_keywords", "")

        class _Safe(dict):
            def __missing__(self, key: str) -> str:
                return "{" + key + "}"

        try:
            return template.format_map(_Safe(merged))
        except (ValueError, IndexError, KeyError):
            # 模板本身有语法问题（例如漏写双花括号）就原样返回，交给模型自己扛
            return template

    def _extra(self) -> str:
        extra = self.settings.extra_prompt
        if not extra:
            return ""
        return f"\n附加要求（优先遵守）：\n{extra}\n"

    # ------------------------------------------------------------ 判屎
    async def judge(
        self,
        suspicion: Suspicion,
        group_name: str,
        bili_infos: list[BiliInfo] | None = None,
        kind: str = "text",
    ) -> ShitVerdict:
        message = suspicion.message
        content = summarise_segments(message.segments, limit=3000)
        if not content:
            content = "（无文本内容，只有媒体）"

        bili_block = ""
        if bili_infos:
            bili_block = "B站视频信息：\n" + "\n\n".join(
                info.summary() for info in bili_infos
            )

        context = ""
        if self.settings.get("with_context", True):
            rendered = self._render_context(suspicion)
            if rendered:
                context = f"上文（{group_name}此前在聊）：\n{rendered}\n"

        prompts = {
            "group_name": group_name or "未知群",
            "kind": kind,
            "trigger": suspicion.reason or "unknown",
            "content": content,
            "BILI_BLOCK": bili_block,
            "CONTEXT_BLOCK": context,
            "anchors": INTENSITY_ANCHORS,
        }
        template = self._prompt("shit_prompt", DEFAULT_SHIT_PROMPT)
        prompt = self._secure_format(template, prompts) + self._extra()

        image_urls = []
        if self.settings.image_llm:
            image_urls = [
                seg.url
                for seg in message.segments
                if seg.kind == "image" and seg.url.startswith(("http", "file"))
            ][: self.settings.max_images_per_message]

        data = await self.llm.generate_json(
            prompt,
            system_prompt=SYSTEM,
            image_urls=image_urls,
            provider_id=str(self.settings.get("analysis_provider_id") or ""),
            umo=message.group_key,
        )
        if data is None:
            return ShitVerdict(analysis=Analysis(), raw={}, ok=False)
        return ShitVerdict(analysis=parse_analysis(data), raw=data, ok=True)

    def _render_context(self, suspicion: Suspicion) -> str:
        """只取最近若干条，避免提示词过长。"""
        limit = int(self.settings.get("max_context_messages") or 6)
        items = suspicion.context_before[-limit:]

        lines = []
        for item in items:
            marks = []
            if item.image_count:
                marks.append(f"{item.image_count}张图")
            if item.has_bili:
                marks.append("B站")
            suffix = f"（{'，'.join(marks)}）" if marks else ""
            lines.append(f"{item.sender_name}：{item.text[:300]}{suffix}")
        return "\n".join(lines)

    # ------------------------------------------------------------ 上下文关系
    async def relate(
        self, messages_text: str, umo: str = ""
    ) -> dict[str, Any] | None:
        if not messages_text.strip():
            return None
        template = self._prompt("context_prompt", DEFAULT_CONTEXT_PROMPT)
        prompt = self._secure_format(
            template,
            {"messages": messages_text, "anchors": INTENSITY_ANCHORS},
        ) + self._extra()
        return await self.llm.generate_json(
            prompt,
            system_prompt=SYSTEM,
            provider_id=str(self.settings.get("analysis_provider_id") or ""),
            umo=umo,
        )

    # ------------------------------------------------------------ 安全
    async def safety_check(self, message: Any, umo: str = "") -> bool | None:
        """强制搬运图片前的安全判定。

        正常流水线会复用 ``judge`` 返回的 ``minor_sexual``；``/shit relay``
        这类调试入口不经过判定，所以单独补一次视觉安全检查。返回 ``None``
        表示无法确认（没有视觉模型、没有图片 URL、模型不可用），调用方必须
        按拦截处理，不能放行。
        """
        image_urls = [
            seg.url
            for seg in message.segments
            if seg.kind == "image" and seg.url.startswith(("http", "file"))
        ][: self.settings.max_images_per_message]
        if not image_urls:
            return None

        data = await self.llm.generate_json(
            DEFAULT_SAFETY_PROMPT,
            system_prompt=SYSTEM,
            image_urls=image_urls,
            provider_id=str(self.settings.get("analysis_provider_id") or ""),
            umo=umo,
        )
        if data is None:
            return None
        value = data.get("minor_sexual")
        if value is None or (
            isinstance(value, str)
            and value.strip().lower() in ("null", "none", "unknown")
        ):
            # 模型明确表示看不清，或没有返回该字段：按“无法确认”处理。
            return None
        return as_bool(value)

    # ------------------------------------------------------------ 群契合度
    async def score_groups(
        self,
        shit_text: str,
        shit_keywords: list[str],
        groups: list[GroupProfile],
        recent_keywords: dict[str, list[str]] | None = None,
        umo: str = "",
    ) -> dict[str, float]:
        """一次性给多个候选群打分，省调用次数。"""
        if not groups:
            return {}
        recent_keywords = recent_keywords or {}

        blocks = []
        for group in groups:
            blocks.append(
                "目标群：{name}\n群标识：{key}\n口味标签：{intake}\n最近吃过：{recent}".format(
                    name=group.name or group.group_key,
                    key=group.group_key,
                    intake="、".join(
                        f"{tag}({weight:.2f})"
                        for tag, weight in sorted(
                            group.intake.items(), key=lambda kv: kv[1], reverse=True
                        )[:8]
                    )
                    or "（还没有口味，样本不足）",
                    recent="、".join(recent_keywords.get(group.group_key, [])[:6])
                    or "（无）",
                )
            )

        template = self._prompt("group_prompt", DEFAULT_GROUP_PROMPT)
        prompt = (
            self._secure_format(
                template,
                {
                    "group_name": "、".join(
                        group.name or group.group_key for group in groups
                    ),
                    "intake": "见下方各群",
                    "recent_keywords": "见下方各群",
                    "shit_text": shit_text[:2000],
                    "shit_keywords": "、".join(shit_keywords[:10]),
                },
            )
            + "\n\n各候选群：\n"
            + "\n\n".join(blocks)
            + "\n\n请在 scores 里用上面的“群标识”作键，给每个群一个 0.0-1.0 的分数。"
            + self._extra()
        )

        data = await self.llm.generate_json(
            prompt,
            system_prompt=SYSTEM,
            provider_id=str(self.settings.get("analysis_provider_id") or ""),
            umo=umo,
        )
        if not data:
            return {}

        raw_scores = data.get("scores") or data
        result: dict[str, float] = {}
        if isinstance(raw_scores, dict):
            for group in groups:
                value = raw_scores.get(group.group_key)
                if value is None:
                    value = raw_scores.get(group.name or "")
                try:
                    result[group.group_key] = max(0.0, min(1.0, float(value)))
                except (TypeError, ValueError):
                    continue
        return result


def as_bool(value: Any, default: bool = False) -> bool:
    """把模型常见的真假写法转成 bool。

    ``bool("false")`` 是 True —— 如果模型把 ``minor_sexual`` 写成字符串
    ``"false"``，直接 bool() 会把正常内容误判成安全拦截。这里只把明确的
    真值字符串当成 True，其余交给 default。
    """
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return value != 0
    if isinstance(value, str):
        text = value.strip().lower()
        if text in ("true", "1", "yes", "y", "on", "是"):
            return True
        if text in ("false", "0", "no", "n", "off", "否", ""):
            return False
    return default


def parse_analysis(data: dict[str, Any]) -> Analysis:
    def as_float(key: str, default: float = 0.0) -> float:
        try:
            return max(0.0, min(1.0, float(data.get(key, default))))
        except (TypeError, ValueError):
            return default

    def as_list(key: str) -> list[str]:
        value = data.get(key) or []
        if isinstance(value, str) or not isinstance(value, (list, tuple, set)):
            value = [value]
        return [str(item).strip() for item in value if str(item).strip()][:12]

    return Analysis(
        is_shit=as_bool(data.get("is_shit", False)),
        intensity=as_float("intensity"),
        tags=as_list("tags"),
        keywords=as_list("keywords"),
        reason=str(data.get("reason", ""))[:120],
        context_relevant=as_bool(data.get("context_relevant", False)),
        context_keywords=as_list("context_keywords"),
        topic_seed=str(data.get("topic_seed", ""))[:200],
        image_tags=as_list("image_tags"),
        characters=as_list("characters"),
        copyrights=as_list("copyrights"),
        nsfw=as_bool(data.get("nsfw", False)),
        minor_sexual=as_bool(data.get("minor_sexual", False)),
    )
