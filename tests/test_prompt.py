"""Prompt calibration tests.

The bug these guard against: intensity had no calibration anchors - only
"0.0 到 1.0 的离谱程度" - so the model returned conservative values
(0.2-0.4 for things that were plainly worth forwarding), and those were
silently dropped by shit_min_intensity.

Also guards the {anchors} placeholder itself: it is injected via
str.format, so a template that references it while the caller forgets to
pass it raises KeyError and breaks every judgement.
"""

from __future__ import annotations

import asyncio
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.accumulator import Suspicion  # noqa: E402
from core.analyzer import (  # noqa: E402
    DEFAULT_CONTEXT_PROMPT,
    DEFAULT_SHIT_PROMPT,
    INTENSITY_ANCHORS,
    SYSTEM,
    Analyzer,
)
from core.config import DEFAULTS, Settings  # noqa: E402
from core.llm import LLMClient  # noqa: E402
from core.models import Message, Segment  # noqa: E402


class RecordingContext:
    def __init__(self, reply: str = '{"is_shit": true, "intensity": 0.55}'):
        self.reply = reply
        self.prompts: list[str] = []
        self.system_prompts: list[str] = []

    async def get_using_provider_async(self, umo=None, *, locale="zh-CN"):
        return type("P", (), {"provider_config": {"id": "p"}})()

    async def llm_generate(self, *, chat_provider_id, prompt=None, system_prompt=None, **kw):
        self.prompts.append(prompt or "")
        self.system_prompts.append(system_prompt or "")
        return type("R", (), {"completion_text": self.reply})()


def make_suspicion() -> Suspicion:
    message = Message(
        group_key="g:Group:1",
        sender_name="群友",
        segments=[Segment(kind="plain", text="这是一段用来测试的群聊内容")],
    )
    return Suspicion(message=message, reason="测试")


class TestAnchorsPresent(unittest.TestCase):
    def test_anchor_block_has_all_bands(self):
        for band in ("0.0-0.2", "0.3-0.4", "0.5-0.6", "0.7-0.8", "0.9-1.0"):
            self.assertIn(band, INTENSITY_ANCHORS)

    def test_shit_prompt_embeds_anchors_placeholder(self):
        self.assertIn("{anchors}", DEFAULT_SHIT_PROMPT)

    def test_context_prompt_embeds_anchors_placeholder(self):
        self.assertIn("{anchors}", DEFAULT_CONTEXT_PROMPT)

    def test_system_prompt_encourages_decisive_scoring(self):
        self.assertIn("锚点", SYSTEM)
        self.assertIn("大胆", SYSTEM)

    # ------------------------------------------------------------ 渲染
    def test_judge_prompt_contains_rendered_anchors(self):
        context = RecordingContext()
        analyzer = Analyzer(Settings(dict(DEFAULTS)), LLMClient(context))
        asyncio.run(analyzer.judge(make_suspicion(), "测试群"))

        prompt = context.prompts[0]
        self.assertIn("0.5-0.6", prompt)
        self.assertIn("0.9-1.0", prompt)
        self.assertNotIn("{anchors}", prompt)  # 没被替换就是 format 漏了

    def test_relate_prompt_contains_rendered_anchors(self):
        context = RecordingContext()
        analyzer = Analyzer(Settings(dict(DEFAULTS)), LLMClient(context))
        asyncio.run(analyzer.relate("甲：你好\n乙：你也好"))

        self.assertIn("0.5-0.6", context.prompts[0])
        self.assertNotIn("{anchors}", context.prompts[0])

    def test_custom_prompt_can_use_anchors(self):
        raw = dict(DEFAULTS)
        raw["shit_prompt"] = "自定义：{content}\n锚点如下\n{anchors}"
        context = RecordingContext()
        analyzer = Analyzer(Settings(raw), LLMClient(context))
        asyncio.run(analyzer.judge(make_suspicion(), "测试群"))

        self.assertIn("自定义", context.prompts[0])
        self.assertIn("0.7-0.8", context.prompts[0])

    def test_custom_prompt_without_anchors_still_works(self):
        """老的自定义提示词里没有 {anchors}，不能因此报错。"""
        raw = dict(DEFAULTS)
        raw["shit_prompt"] = "只看内容：{content}"
        context = RecordingContext()
        analyzer = Analyzer(Settings(raw), LLMClient(context))
        verdict = asyncio.run(analyzer.judge(make_suspicion(), "测试群"))

        self.assertTrue(verdict.ok)


class TestThreshold(unittest.TestCase):
    def test_default_threshold_lowered(self):
        self.assertEqual(DEFAULTS["shit_min_intensity"], 0.45)

    def test_threshold_sits_below_share_band(self):
        """0.45 应该低于“值得跨群分享”的 0.5，否则锚点白给。"""
        self.assertLess(DEFAULTS["shit_min_intensity"], 0.5)


if __name__ == "__main__":
    unittest.main()
