"""B站信息：识别 + 抓取。

识别永远是本插件自己的（正则，零依赖）。抓取有两种模式：

1. 软依赖 ``astrbot_plugin_bilibili``：能 import 到就跑它上面挂的
   ``BiliUserDynamicsTool`` / 内部 client —— 实际上那一堆 tool 都是给 LLM 用的，
   我们直接调它的 ``BiliClient.get_video_info(bvid=...)`` 更直接；
2. 兜底：自己打官方开放接口 ``x/web-interface/view``，不依赖任何插件。

两条路都可能因为网络/风控失败，所以失败时返回降级信息（只有 BV 号），
不阻塞主流程。
"""

from __future__ import annotations

import importlib
import json
import re
from dataclasses import dataclass, field
from typing import Any

# BV 号：BV 开头 + 10 位
BV_RE = re.compile(r"(BV[0-9A-Za-z]{10})", re.IGNORECASE)
# b23 短链
B23_RE = re.compile(r"https?://b23\.tv/[0-9A-Za-z]+", re.IGNORECASE)
# 视频页
VIDEO_URL_RE = re.compile(
    r"https?://(?:www\.)?bilibili\.com/video/(BV[0-9A-Za-z]{10})", re.IGNORECASE
)
# 分享文本里的标题：“【xxx】”
SHARE_TITLE_RE = re.compile(r"【([^】]{2,80})】")

BILI_VIEW_API = "https://api.bilibili.com/x/web-interface/view?bvid={bvid}"
BILI_UA = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/124.0 Safari/537.36"
)


@dataclass(slots=True)
class BiliRefs:
    bvids: list[str] = field(default_factory=list)
    short_urls: list[str] = field(default_factory=list)
    shares: list[str] = field(default_factory=list)

    def any(self) -> bool:
        return bool(self.bvids or self.short_urls or self.shares)


def extract_bili_ids(text: str) -> BiliRefs:
    refs = BiliRefs()
    if not text:
        return refs

    for match in VIDEO_URL_RE.finditer(text):
        refs.bvids.append(match.group(1))
    for match in BV_RE.finditer(text):
        if match.group(1) not in refs.bvids:
            refs.bvids.append(match.group(1))
    refs.short_urls = [url for url in B23_RE.findall(text)]

    for title in SHARE_TITLE_RE.findall(text):
        if "bilibili" in text.lower() or "b站" in text.lower():
            refs.shares.append(title)
    return refs


@dataclass(slots=True)
class BiliInfo:
    bvid: str
    title: str = ""
    owner: str = ""
    desc: str = ""
    view: int = 0
    like: int = 0
    duration: int = 0
    tags: list[str] = field(default_factory=list)
    source: str = "none"  # astrbot_plugin_bilibili | direct | fallback

    def summary(self) -> str:
        if not self.title:
            return f"B站视频 {self.bvid}（未取到详细信息）"
        bits = [f"标题：{self.title}"]
        if self.owner:
            bits.append(f"UP：{self.owner}")
        if self.duration:
            bits.append(f"时长：{self.duration // 60}分{self.duration % 60}秒")
        if self.view:
            bits.append(f"播放：{self.view}")
        if self.desc:
            bits.append(f"简介：{self.desc[:120]}")
        return "\n".join(bits)


class BiliResolver:
    """按“软依赖 → 直连 → 降级”的顺序取视频信息。"""

    def __init__(self, proxy: str = "", timeout: float = 6.0) -> None:
        self.proxy = proxy
        self.timeout = timeout
        self._plugin_client: Any = None
        self._plugin_checked = False

    # ---------------------------------------------------------------- 软依赖
    def _find_plugin_client(self) -> Any:
        """在 AstrBot 的 Star 管理器里找到 bilibili 插件实例。"""
        if self._plugin_checked:
            return self._plugin_client
        self._plugin_checked = True
        try:
            from astrbot.core.star.star_manager import (
                star_map,
            )  # type: ignore[import-not-found]
        except Exception:  # pragma: no cover - 取决于 AstrBot 版本
            star_map = {}
        for name, instance in (star_map or {}).items():  # pragma: no cover
            if "bilibili" in str(name) and hasattr(instance, "bili_client"):
                self._plugin_client = instance.bili_client
                break
        return self._plugin_client

    async def _via_plugin(self, bvid: str) -> BiliInfo | None:
        client = self._find_plugin_client()
        if client is None:
            return None
        try:  # pragma: no cover - 需要真实插件
            data = await client.get_video_info(bvid=bvid)
        except Exception:
            return None
        if not data:
            return None
        info = data.get("info") or {}
        return BiliInfo(
            bvid=bvid,
            title=str(info.get("title", "")),
            owner=str((info.get("owner") or {}).get("name", "")),
            desc=str(info.get("desc", ""))[:400],
            view=int((info.get("stat") or {}).get("view", 0) or 0),
            like=int((info.get("stat") or {}).get("like", 0) or 0),
            duration=int(info.get("duration", 0) or 0),
            source="astrbot_plugin_bilibili",
        )

    # ---------------------------------------------------------------- 直连
    async def _via_direct(self, bvid: str) -> BiliInfo | None:
        try:
            import aiohttp  # type: ignore[import-not-found]
        except Exception:
            return None
        try:  # pragma: no cover - 需要网络
            timeout = aiohttp.ClientTimeout(total=self.timeout)
            async with aiohttp.ClientSession(timeout=timeout) as session:
                async with session.get(
                    BILI_VIEW_API.format(bvid=bvid),
                    headers={"User-Agent": BILI_UA, "Referer": "https://www.bilibili.com/"},
                    proxy=self.proxy or None,
                ) as resp:
                    payload = await resp.json(content_type=None)
        except Exception:
            return None
        if not isinstance(payload, dict) or payload.get("code") != 0:
            return None
        data = payload.get("data") or {}
        return BiliInfo(
            bvid=bvid,
            title=str(data.get("title", "")),
            owner=str((data.get("owner") or {}).get("name", "")),
            desc=str(data.get("desc", ""))[:400],
            view=int((data.get("stat") or {}).get("view", 0) or 0),
            like=int((data.get("stat") or {}).get("like", 0) or 0),
            duration=int(data.get("duration", 0) or 0),
            source="direct",
        )

    async def resolve(self, text: str, allow_network: bool = True) -> list[BiliInfo]:
        refs = extract_bili_ids(text)
        results: list[BiliInfo] = []

        for bvid in refs.bvids[:2]:
            info = await self._via_plugin(bvid)
            if info is None and allow_network:
                info = await self._via_direct(bvid)
            if info is None:
                info = BiliInfo(bvid=bvid, source="fallback")
            results.append(info)

        if not results and refs.short_urls:
            # 短链需要先解析重定向；这里不额外发请求，交给 LLM 从卡片文字判断
            results.append(
                BiliInfo(bvid="", desc=refs.short_urls[0], source="fallback")
            )

        return results


def bili_card_hint(json_text: str) -> str:
    """从 QQ 小程序卡片 JSON 里抠出标题与跳转链接，给 LLM 当上下文。"""
    try:
        data = json.loads(json_text)
    except Exception:
        return ""
    meta = (data.get("meta") or {}) if isinstance(data, dict) else {}
    detail = meta.get("detail_1") or {}
    news = meta.get("news") or {}
    title = detail.get("title") or news.get("title") or meta.get("title") or ""
    desc = detail.get("desc") or news.get("desc") or ""
    url = detail.get("qqdocurl") or news.get("jumpUrl") or ""
    if not (title or url):
        return ""
    return "B站分享卡片：" + " | ".join(x for x in (str(title), str(desc), str(url)) if x)
