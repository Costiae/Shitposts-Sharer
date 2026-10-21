"""AstrBot LLM 调用封装：文本 + 多模态 + JSON 解析。

只依赖 ``context.llm_generate(...)``，不引入任何第三方 SDK。
调用失败一律返回 ``None``，由上层决定是放过还是跳过 —— 绝不因为模型抽风
就让插件崩掉或刷屏。
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any

_FENCE = re.compile(r"```(?:json)?\s*(.*?)```", re.DOTALL)
_TRAILING_COMMA = re.compile(r",\s*([}\]])")


@dataclass(slots=True)
class LLMResult:
    text: str
    provider_id: str = ""
    error: str = ""


def extract_json(text: str) -> dict[str, Any] | None:
    """从模型输出里抠出第一个 JSON 对象，容忍 ```json 围栏与尾逗号。"""
    if not text:
        return None
    candidates: list[str] = []
    fenced = _FENCE.search(text)
    if fenced:
        candidates.append(fenced.group(1))
    candidates.append(text)

    for candidate in candidates:
        candidate = candidate.strip()
        for start in range(len(candidate)):
            if candidate[start] != "{":
                continue
            depth = 0
            in_string = False
            escaped = False
            for index in range(start, len(candidate)):
                char = candidate[index]
                if in_string:
                    if escaped:
                        escaped = False
                    elif char == "\\":
                        escaped = True
                    elif char == '"':
                        in_string = False
                    continue
                if char == '"':
                    in_string = True
                elif char == "{":
                    depth += 1
                elif char == "}":
                    depth -= 1
                    if depth == 0:
                        blob = candidate[start : index + 1]
                        for attempt in (blob, _TRAILING_COMMA.sub(r"\1", blob)):
                            try:
                                parsed = json.loads(attempt)
                            except json.JSONDecodeError:
                                continue
                            if isinstance(parsed, dict):
                                return parsed
                        break
    return None


class LLMClient:
    """薄封装，方便在测试里替换成假的 context。"""

    def __init__(self, context: Any) -> None:
        self.context = context

    @property
    def available(self) -> bool:
        return hasattr(self.context, "llm_generate")

    async def generate(
        self,
        prompt: str,
        *,
        system_prompt: str = "",
        image_urls: list[str] | None = None,
        provider_id: str = "",
    ) -> LLMResult:
        if not self.available:
            return LLMResult(text="", error="llm_generate 不可用")

        kwargs: dict[str, Any] = {"prompt": prompt}
        if system_prompt:
            kwargs["system_prompt"] = system_prompt
        if image_urls:
            # AstrBot 的 llm_generate 接受 image_url 列表做多模态
            kwargs["image_urls"] = image_urls
        if provider_id:
            kwargs["chat_provider_id"] = provider_id

        try:
            response = await self.context.llm_generate(**kwargs)
        except TypeError:
            # 老的/不同的签名：退回纯文本，丢掉图片参数
            try:
                response = await self.context.llm_generate(prompt=prompt)
            except Exception as exc:  # pragma: no cover
                return LLMResult(text="", error=f"{type(exc).__name__}: {exc}")
        except Exception as exc:  # pragma: no cover
            return LLMResult(text="", error=f"{type(exc).__name__}: {exc}")

        text = getattr(response, "completion_text", None)
        if text is None:
            text = str(response or "")
        return LLMResult(text=str(text), provider_id=provider_id)

    async def generate_json(
        self,
        prompt: str,
        *,
        system_prompt: str = "",
        image_urls: list[str] | None = None,
        provider_id: str = "",
    ) -> dict[str, Any] | None:
        result = await self.generate(
            prompt,
            system_prompt=system_prompt,
            image_urls=image_urls,
            provider_id=provider_id,
        )
        return extract_json(result.text)
