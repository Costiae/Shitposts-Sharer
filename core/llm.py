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


def provider_id_of(provider: Any) -> str:
    """从 AstrBot Provider 实例里取出它的 id。

    AstrBot 把 id 放在 ``provider_config["id"]`` 里；不同版本可能只把
    它挂在 ``id`` 属性上，所以两条路都试。
    """
    if provider is None:
        return ""
    config = getattr(provider, "provider_config", None)
    if isinstance(config, dict):
        value = config.get("id")
        if value:
            return str(value)
    for attr in ("id", "provider_id"):
        value = getattr(provider, attr, None)
        if value:
            return str(value)
    return ""


class LLMClient:
    """薄封装，方便在测试里替换成假的 context。"""

    def __init__(self, context: Any) -> None:
        self.context = context

    @property
    def available(self) -> bool:
        return hasattr(self.context, "llm_generate")

    async def _resolve_provider_id(self, umo: str = "") -> str:
        """解析“当前对话模型”的 id。

        ``context.llm_generate`` 的 ``chat_provider_id`` 是必填的
        keyword-only 参数（见 AstrBot ``astrbot/core/star/context.py``）：

            async def llm_generate(self, *, chat_provider_id: str, ...)

        配置留空时如果不显式解析一个 id，就会抛
        ``TypeError: missing required keyword argument 'chat_provider_id'``，
        被下面的兜底吞成空文本，最终表现为“模型输出无法解析”。
        """
        # 4.29 起同步版已 deprecated，优先用异步版
        getter = getattr(self.context, "get_using_provider_async", None)
        if callable(getter):
            for call in (lambda: getter(umo=umo or None), getter):
                try:
                    provider = call()
                    if hasattr(provider, "__await__"):
                        provider = await provider
                except (TypeError, ValueError):
                    continue
                except Exception:
                    continue
                resolved = provider_id_of(provider)
                if resolved:
                    return resolved
            return ""

        # 老版本只有同步版
        getter = getattr(self.context, "get_using_provider", None)
        if callable(getter):
            for call in (lambda: getter(umo=umo or None), lambda: getter()):
                try:
                    provider = call()
                except (TypeError, ValueError):
                    continue
                except Exception:
                    continue
                resolved = provider_id_of(provider)
                if resolved:
                    return resolved
        return ""

    async def generate(
        self,
        prompt: str,
        *,
        system_prompt: str = "",
        image_urls: list[str] | None = None,
        provider_id: str = "",
        umo: str = "",
    ) -> LLMResult:
        if not self.available:
            return LLMResult(text="", error="llm_generate 不可用")

        # chat_provider_id 是必填参数，留空时必须先解析出一个真实 id
        resolved = provider_id or await self._resolve_provider_id(umo)
        if not resolved:
            return LLMResult(
                text="",
                error=(
                    "没有可用的对话模型：请在 AstrBot 里启用一个 chat provider，"
                    "或在本插件配置里填 analysis_provider_id"
                ),
            )

        kwargs: dict[str, Any] = {
            "chat_provider_id": resolved,
            "prompt": prompt,
        }
        if system_prompt:
            kwargs["system_prompt"] = system_prompt
        if image_urls:
            kwargs["image_urls"] = image_urls

        try:
            response = await self.context.llm_generate(**kwargs)
        except TypeError as exc:
            return LLMResult(
                text="",
                error=f"llm_generate 签名不匹配: {type(exc).__name__}: {exc}",
            )
        except Exception as exc:  # pragma: no cover - 网络/额度等
            return LLMResult(text="", error=f"{type(exc).__name__}: {exc}")

        text = getattr(response, "completion_text", None)
        if text is None:
            text = str(response or "")
        return LLMResult(text=str(text), provider_id=resolved)

    async def generate_json(
        self,
        prompt: str,
        *,
        system_prompt: str = "",
        image_urls: list[str] | None = None,
        provider_id: str = "",
        umo: str = "",
    ) -> dict[str, Any] | None:
        result = await self.generate(
            prompt,
            system_prompt=system_prompt,
            image_urls=image_urls,
            provider_id=provider_id,
            umo=umo,
        )
        return extract_json(result.text)
