"""LLM client tests.

The bug these guard against: AstrBot's llm_generate declares
``chat_provider_id`` as a required keyword-only argument

    async def llm_generate(self, *, chat_provider_id: str, prompt=None, ...)

so omitting it when the plugin's analysis_provider_id is empty raises
TypeError. The old code swallowed every exception into LLMResult(text=""),
which surfaced downstream as the useless message "模型输出无法解析".
"""

from __future__ import annotations

import asyncio
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.llm import LLMClient, LLMResult, extract_json, provider_id_of  # noqa: E402


class FakeProvider:
    """Mirrors AstrBot's Provider: id lives in provider_config."""

    def __init__(self, provider_id: str):
        self.provider_config = {"id": provider_id, "type": "openai_chat_completion"}
        self.model_name = "gpt-4o-mini"


class FakeResponse:
    def __init__(self, text: str):
        self.completion_text = text


class StrictContext:
    """Same signature as AstrBot 4.29.0 - chat_provider_id is required."""

    def __init__(self, provider=None, text='{"is_shit": true}'):
        self._provider = provider
        self.text = text
        self.calls: list[dict] = []

    async def get_using_provider_async(self, umo=None, *, locale="zh-CN"):
        return self._provider

    async def llm_generate(self, *, chat_provider_id: str, prompt=None, **kwargs):
        self.calls.append(
            {"chat_provider_id": chat_provider_id, "prompt": prompt, **kwargs}
        )
        if not chat_provider_id:
            raise TypeError("chat_provider_id must be a non-empty string")
        return FakeResponse(self.text)


class TestProviderResolution(unittest.TestCase):
    def test_provider_id_from_config(self):
        self.assertEqual(provider_id_of(FakeProvider("deepseek")), "deepseek")

    def test_provider_id_falls_back_to_attribute(self):
        class Bare:
            id = "attr-id"

        self.assertEqual(provider_id_of(Bare()), "attr-id")

    def test_provider_id_none(self):
        self.assertEqual(provider_id_of(None), "")

    # ------------------------------------------------------------ 回归
    def test_empty_config_still_supplies_chat_provider_id(self):
        """留空 analysis_provider_id 时必须自动解析，而不是抛 TypeError。"""
        context = StrictContext(provider=FakeProvider("auto-provider"))
        client = LLMClient(context)

        result = asyncio.run(client.generate("hello", provider_id=""))

        self.assertEqual(result.error, "", f"unexpected error: {result.error}")
        self.assertEqual(result.provider_id, "auto-provider")
        self.assertEqual(context.calls[0]["chat_provider_id"], "auto-provider")
        self.assertEqual(result.text, '{"is_shit": true}')

    def test_explicit_provider_id_wins(self):
        context = StrictContext(provider=FakeProvider("auto-provider"))
        client = LLMClient(context)

        asyncio.run(client.generate("hi", provider_id="explicit"))

        self.assertEqual(context.calls[0]["chat_provider_id"], "explicit")

    def test_no_provider_returns_actionable_error(self):
        """没有可用模型时，错误信息要能指向解决办法，不能是空字符串。"""
        context = StrictContext(provider=None)
        client = LLMClient(context)

        result = asyncio.run(client.generate("hi"))

        self.assertEqual(result.text, "")
        self.assertIn("chat provider", result.error)
        self.assertIn("analysis_provider_id", result.error)
        self.assertEqual(context.calls, [])  # 不该发出注定失败的调用

    def test_umo_is_forwarded_for_session_isolation(self):
        context = StrictContext(provider=FakeProvider("p"))
        client = LLMClient(context)
        seen = {}

        original = context.get_using_provider_async

        async def spy(umo=None, *, locale="zh-CN"):
            seen["umo"] = umo
            return await original(umo=umo, locale=locale)

        context.get_using_provider_async = spy
        asyncio.run(client.generate("hi", umo="aiocqhttp:GroupMessage:123"))

        self.assertEqual(seen["umo"], "aiocqhttp:GroupMessage:123")

    def test_legacy_sync_provider_lookup(self):
        class SyncContext:
            def get_using_provider(self, umo=None):
                return FakeProvider("sync-provider")

            async def llm_generate(self, *, chat_provider_id: str, prompt=None, **kw):
                return FakeResponse("{}")

        client = LLMClient(SyncContext())
        self.assertEqual(
            asyncio.run(client._resolve_provider_id("")), "sync-provider"
        )

    def test_signature_mismatch_reports_typeerror_not_empty(self):
        class Weird:
            async def llm_generate(self, prompt=None):
                return FakeResponse("{}")

            async def get_using_provider_async(self, umo=None, *, locale="zh-CN"):
                return FakeProvider("p")

        result = asyncio.run(LLMClient(Weird()).generate("hi"))
        self.assertEqual(result.text, "")
        self.assertIn("llm_generate", result.error)

    def test_image_urls_still_forwarded(self):
        context = StrictContext(provider=FakeProvider("p"))
        asyncio.run(
            LLMClient(context).generate(
                "描述这张图", image_urls=["https://example.com/a.png"]
            )
        )
        self.assertEqual(
            context.calls[0]["image_urls"], ["https://example.com/a.png"]
        )


class TestExtractJson(unittest.TestCase):
    def test_plain(self):
        self.assertEqual(extract_json('{"a": 1}'), {"a": 1})

    def test_fenced(self):
        self.assertEqual(
            extract_json('好的\n```json\n{"a": 1}\n```'), {"a": 1}
        )

    def test_empty_returns_none(self):
        self.assertIsNone(extract_json(""))


if __name__ == "__main__":
    unittest.main()
