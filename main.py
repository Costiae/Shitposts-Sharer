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
from .core.detector import message_from_event
from .core.pipeline import Loggerish, Pipeline
from .core.storage import Storage
from .core.tracelog import Trace, describe_message

# 插件指令一律 ASCII；alias 只作为历史中文指令的兼容入口。
# ``sps`` = ShitPosts Sharer：QQ 的 emoji 输入会把 /shit 当成表情搜索，
# 这个无意义短串不会被表情候选吃掉。
COMMAND = "shit"
ALIASES = {"spam", "ss", "sps"}


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
        self.pipeline = Pipeline(
            context=self.context,
            settings=self.settings,
            storage=self.storage,
            logger=Loggerish(logger),
        )
        # 调试指令要读运行时状态，通过探针注入而不是让指令层依赖 pipeline
        self.commands = Commands(
            self.storage,
            self.settings,
            probe_llm=self.pipeline.llm,
            probe_pipeline=self.pipeline,
            probe_force=self.pipeline,
        )
        logger.info(f"Shitposts Sharer loaded, db: {self.db_path}")

    async def initialize(self) -> None:
        await self.pipeline.start()

    # ------------------------------------------------------------------ 监听
    @filter.event_message_type(filter.EventMessageType.ALL)
    async def on_any_message(self, event: AstrMessageEvent):
        """所有消息都过一遍。

        EARLY 日志是整条链路的第一层：它证明 AstrBot 确实把消息交给了插件。
        如果群里发了消息但这里**没有**对应日志，说明消息在到插件之前就被
        AstrBot 拦掉了（未唤醒 / 白名单 / 平台没上报），那就不是插件的问题。
        """
        early = Trace(Loggerish(logger), str(self.settings.get("log_level") or "normal"))
        early.gate("entry")

        if not is_group_event(event):
            early.skip("not a group message")
            return

        try:
            message = message_from_event(
                event, str(getattr(event, "unified_msg_origin", "") or "")
            )
            early.step(
                f"EARLY {describe_message(message, limit=200)} "
                f"wake={getattr(event, 'is_wake', '?')} "
                f"at_or_wake={getattr(event, 'is_at_or_wake_command', '?')}"
            )
            await self.pipeline.on_event(event)
        except Exception as exc:  # pragma: no cover - 绝不让异常冒到框架
            early.error(f"listener crashed: {type(exc).__name__}: {exc}")
            logger.error(
                f"[shit] entry | listener crashed: {type(exc).__name__}: {exc}"
            )

    # ------------------------------------------------------------------ 指令
    #
    # SIGNATURE RULE: command handlers take (self, event) and nothing else.
    #
    # AstrBot's CommandFilter.init_handler_md() records every parameter after
    # `self` and `event`, then validate_and_convert_params() CALLS that value
    # as a converter:
    #
    #     result[param_name] = param_type_or_default_val(params[i])
    #
    # A trailing *args is recorded as inspect.Parameter.empty, so it becomes
    # `_empty("help")` -> TypeError: _empty() takes no arguments. The caller
    # only catches ValueError, so the TypeError reaches waking_check, which
    # posts an error into the group and calls stop_event() - the command is
    # silently swallowed and never runs.
    #
    # Any per-command arguments are therefore parsed out of event.message_str
    # by hand. Never add *args, **kwargs or a typed positional parameter here.
    @filter.command(COMMAND, alias=ALIASES)
    async def shit(self, event: AstrMessageEvent):
        """/shit <subcommand>: register, status, threshold, ...

        NOTE: this handler must never let its own exceptions escape either.
        AstrBot turns an uncaught exception into an error traceback posted
        straight into the group, which is exactly how a plugin looks broken.
        """
        parts = _command_args(event, {COMMAND, *ALIASES})

        try:
            result = self.commands.dispatch(
                event, parts, is_admin=_is_admin(event)
            )
            text = result.text
        except Exception as exc:  # pragma: no cover - defensive
            logger.error(f"shit command failed: {type(exc).__name__}: {exc}")
            text = "Command failed, see the AstrBot log."

        yield event.plain_result(text or "(empty)")

    @filter.command("shoot", alias={"manual"})
    async def manual_shot(self, event: AstrMessageEvent):
        """/shoot <text>: admin-only, push a text straight into the pipeline.

        No *args - see the signature rule above.
        """
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


def _command_args(event: AstrMessageEvent, names: set[str]) -> list[str]:
    """从原始消息里取出指令后的参数。

    不使用装饰器注入的参数，原因见 ``shit`` 上方的 SIGNATURE RULE：
    任何额外的形参（包括 ``*args``）都会被 AstrBot 的参数解析器当作
    转换器调用，``*args`` 对应 ``inspect.Parameter.empty``，直接抛
    ``TypeError: _empty() takes no arguments``，异常一路上抛让指令被
    ``stop_event()`` 吞掉。所以参数一律从 ``message_str`` 自己切，
    顺带也支持别名。
    """
    text = (getattr(event, "message_str", "") or "").strip()
    parts = text.split()
    if parts and _strip_slash(parts[0]) in names:
        parts = parts[1:]
    return [part for part in parts if part]


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
