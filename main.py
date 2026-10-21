"""Shitposts Sharer —— AstrBot 自动化搬屎插件。

职责只有三件事：
1. 监听已注册群的**全部**消息，攒上下文；
2. 命中候选就丢给多模态 LLM 判断"是不是屎"，并提炼话题关键词；
3. 把屎搬到口味匹配、可信度够格、方向允许的群。

具体逻辑都在 core/ 下，这里只做 AstrBot 胶水。
"""

from __future__ import annotations

from pathlib import Path

from astrbot.api import AstrBotConfig, logger  # type: ignore[import-not-found]
from astrbot.api.event import AstrMessageEvent, filter  # type: ignore[import-not-found]
from astrbot.api.star import Context, Star, register  # type: ignore[import-not-found]

from .core.commands import Commands, CommandResult, is_group_event
from .core.config import Settings
from .core.pipeline import Loggerish, Pipeline
from .core.storage import Storage

# 插件指令一律 ASCII；alias 只作为历史中文指令的兼容入口
COMMAND = "shit"
ALIASES = {"spam", "ss"}


@register(
    "astrbot_plugin_shitposts_sharer",
    "Costia",
    "Detect group chat shitposts, learn each group's taste, and auto-relay them",
    "0.1.0",
    "https://github.com/Costiae/Shitposts-Sharer",
)
class ShitpostsSharer(Star):
    def __init__(self, context: Context, config: AstrBotConfig) -> None:
        super().__init__(context)
        self.context = context
        self.config = config
        self.settings = Settings(dict(config) if config else {})

        data_dir = _resolve_data_dir()
        data_dir.mkdir(parents=True, exist_ok=True)
        self.db_path = data_dir / "shitposts.sqlite3"

        self.storage = Storage(self.db_path)
        self.commands = Commands(self.storage, self.settings)
        self.pipeline = Pipeline(
            context=self.context,
            settings=self.settings,
            storage=self.storage,
            logger=Loggerish(logger),
        )
        logger.info(f"搬屎插件已加载，数据库：{self.db_path}")

    async def initialize(self) -> None:
        await self.pipeline.start()

    # ------------------------------------------------------------------ 监听
    @filter.event_message_type(filter.EventMessageType.ALL)
    async def on_any_message(self, event: AstrMessageEvent):
        """所有消息都过一遍：已注册群才处理。"""
        if not is_group_event(event):
            return
        try:
            await self.pipeline.on_event(event)
        except Exception as exc:  # pragma: no cover
            logger.error(f"搬屎监听出错：{type(exc).__name__}: {exc}")

    # ------------------------------------------------------------------ 指令
    @filter.command(COMMAND, alias=ALIASES)
    async def shit(self, event: AstrMessageEvent, *args):
        """/shit <subcommand>: register, status, threshold, ..."""
        text = event.message_str.strip()
        parts = text.split()
        # 去掉命令词本身（含中文别名）
        if parts and _strip_slash(parts[0]) in {COMMAND, *ALIASES}:
            parts = parts[1:]

        try:
            result: CommandResult = self.commands.dispatch(
                event, parts, is_admin=_is_admin(event)
            )
        except Exception as exc:  # pragma: no cover
            logger.error(f"shit command failed: {type(exc).__name__}: {exc}")
            yield event.plain_result("Command failed, see the log for details.")
            return

        yield event.plain_result(result.text)

    @filter.command("shoot", alias={"manual"})
    async def manual_shot(self, event: AstrMessageEvent, *args):
        """/shoot <text>: admin-only, push a text straight into the pipeline."""
        if not _is_admin(event):
            yield event.plain_result("Admin only.")
            return

        text = event.message_str.strip()
        parts = text.split(maxsplit=1)
        payload = parts[1] if len(parts) > 1 else ""
        if not payload:
            yield event.plain_result("Usage: /shoot <text>")
            return

        group = self.storage.get_group(event.unified_msg_origin)
        if group is None:
            yield event.plain_result(
                "This group is not registered. Run /shit register both first."
            )
            return

        from .core.models import GroupProfile, Message, Segment

        message = Message(
            group_key=group.group_key,
            sender_name="管理员",
            segments=[Segment(kind="plain", text=payload)],
        )
        suspicion = self.pipeline.accumulator.open(message, "manual")
        result = await self.pipeline.handle_suspicion(suspicion)
        if result is None:
            yield event.plain_result("Not judged as a shit. Check the LLM output in the log.")
            return
        keywords = ", ".join(result.get("keywords", [])) or "none"
        delivered = ", ".join(result.get("delivered", [])) or "no group took it"
        yield event.plain_result(f"keywords: {keywords}\ndelivered: {delivered}")

    async def terminate(self) -> None:
        await self.pipeline.stop()
        self.storage.close()
        logger.info("Shitposts Sharer unloaded.")


def _strip_slash(token: str) -> str:
    return token.lstrip("/").lower()


def _resolve_data_dir() -> Path:
    """优先用 AstrBot 的 data 目录，拿不到就退到插件目录下的 data/。"""
    try:  # pragma: no cover - 取决于 AstrBot 版本
        from astrbot.core.utils.astrbot_path import (  # type: ignore
            get_astrbot_data_path,
        )

        return Path(get_astrbot_data_path()) / "plugin_data" / "shitposts_sharer"
    except Exception:
        return Path(__file__).parent / "data"


def _is_admin(event: AstrMessageEvent) -> bool:
    checker = getattr(event, "is_admin", None)
    try:
        if callable(checker) and checker():
            return True
    except Exception:  # pragma: no cover
        pass
    role = str(getattr(event, "role", "") or "")
    return role in ("admin", "owner", "founder")
