"""群指令：跨群管理搬运网络。

搬屎的接收方是静默的：群里的人不配置"自己这个群怎么收屎"。
实际使用者是运营者 —— 坐在 A 群（或私聊）里操作 B 群。所以每条子指令的
**第一个参数是目标群**，省略时默认当前会话：

    /shit register in           在当前群注册（群友自助用）
    /shit pause 123456          在 A 群把 B 群暂停（管理员）
    /shit on aiocqhttp:GroupMessage:123456
    /shit list                  A 群和 B 群谁该收谁该发

权限：只操作当前会话时所有人都能用；一旦目标是别的群，必须是管理员。
所有指令与回复文本保持 ASCII。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .config import Settings
from .models import Direction, GroupProfile
from .profile import confidence_score, describe_intake
from .storage import Storage
from .targets import (
    current_origin,
    describe_group,
    looks_like_target,
    render_ambiguous,
    resolve,
)

HELP_TEXT = """Shitposts Sharer commands
  <target> = group id, full UMO, #index, or name. Omit = current session.

  /shit register <receive|emit|both> [target] [threshold]   join the network
  /shit unregister [target]                           leave the network
  /shit on <target>                                   enable relay for a group
  /shit off <target>                                  pause relay for a group
  /shit status [target]                               taste + confidence
  /shit list                                          all groups
  /shit threshold <0-1|xx%> [target]                  set delivery threshold
  /shit purge [target]                                delete a group (admin)
  /shit help                                          this text

Debug (admin):
  /shit debug [target]                                show every gate and why it failed
  /shit tags [target]                                 dump taste terms (all groups if omitted)
  /shit relay [target]                                bypass direction/taste/confidence, relay now
  /shit intake [target]                               force this message into the taste update
  /shoot <text>                                       admin: test one payload

Direction describes THE GROUP, not the plugin:
  receive   this group receives shit from other groups; it is never a source
  emit      this group produces shit that gets relayed out; it receives nothing
  both      this group both produces and receives
Handling a group other than the current one requires admin."""


@dataclass(slots=True)
class CommandResult:
    """指令层的纯数据结果。

    刻意不继承、不叫 CommandResult 之外的任何名字：AstrBot 也导出了一个
    ``CommandResult``（= MessageEventResult 的别名）。两者在 isinstance 判定里
    互不相干，幸好我们用 ``from .core.commands import ...`` 显式导入，
    不会串到 AstrBot 那个。
    """

    text: str
    ok: bool = True


class Commands:
    def __init__(
        self,
        storage: Storage,
        settings: Settings,
        *,
        probe_llm: Any = None,
        probe_pipeline: Any = None,
        probe_force: Any = None,
    ) -> None:
        self.storage = storage
        self.settings = settings
        # 可选探针：调试指令要读运行时状态，但指令层不该直接依赖 pipeline。
        # probe_llm        -> 有 .available 属性
        # probe_pipeline   -> 有 .pending() 方法
        # probe_force      -> 有 .run(event, target_key, incoming) 方法
        self.probe_llm = probe_llm
        self.probe_pipeline = probe_pipeline
        self.probe_force = probe_force

    # ------------------------------------------------------------------ 主分发
    def dispatch(
        self, event: Any, args: list[str], *, is_admin: bool = False
    ) -> CommandResult:
        if not args:
            return CommandResult(HELP_TEXT)

        action = args[0].strip().lower()
        rest = args[1:]

        if action in ("register", "reg", "add"):
            return self.register(event, rest, is_admin=is_admin)
        if action in ("unregister", "del", "remove", "rm"):
            return self.unregister(event, rest, is_admin=is_admin)
        if action in ("on", "enable", "resume", "start"):
            return self.toggle(event, rest, enabled=True, is_admin=is_admin)
        if action in ("off", "disable", "pause", "stop"):
            return self.toggle(event, rest, enabled=False, is_admin=is_admin)
        if action in ("status", "info", "show"):
            return self.status(event, rest)
        if action in ("list", "ls", "groups"):
            return self.list_groups()
        if action in ("threshold", "thr"):
            return self.set_threshold(event, rest, is_admin=is_admin)
        if action in ("purge", "drop"):
            return self.purge(event, rest, is_admin=is_admin)
        if action in ("help", "h", "?"):
            return CommandResult(HELP_TEXT)

        # ---- debug surface (admin) ----
        if action in ("debug", "dbg"):
            return self.debug(event, rest, is_admin=is_admin)
        if action in ("tags", "tag"):
            return self.tags(event, rest, is_admin=is_admin)
        if action in ("relay", "forceout", "push"):
            # 当作源屎，忽略方向/口味/可信度，直接推出去
            return self.force_delivery(event, rest, is_admin=is_admin, incoming=True)
        if action in ("intake", "forcein", "absorb"):
            # 当作被搬入的屎，只更新目标群口味，不外发
            return self.force_delivery(event, rest, is_admin=is_admin, incoming=False)

        return CommandResult(f"Unknown subcommand: {action}\n\n{HELP_TEXT}", ok=False)

    # ------------------------------------------------------------------ 调试
    def debug(
        self, event: Any, args: list[str], *, is_admin: bool = False
    ) -> CommandResult:
        """把判定链路里每一道闸门的结果都摊开，专门用来排查“为什么没搬”。"""
        if not is_admin:
            return CommandResult("Admin only.", ok=False)

        token = args[0] if args else ""
        resolution = resolve(self.storage, event, token)
        if not resolution.ok:
            return CommandResult(
                f"Target not found: {token or '(current)'}. "
                f"Use /shit debug inside a registered group.",
                ok=False,
            )
        group = resolution.group
        assert group is not None

        confidence = self._confidence(group)
        threshold = self.settings.threshold_for(group.threshold)
        enabled = bool(self.settings.get("enabled"))
        watch_only = bool(self.settings.get("watch_only_registered"))
        rerank = bool(self.settings.get("llm_rerank"))
        llm_ok = getattr(self.probe_llm, "available", None) if hasattr(self, "probe_llm") else None

        checks = [
            ("plugin enabled", enabled),
            ("group registered", True),
            ("group enabled", group.enabled),
            ("can emit (direction)", group.direction.can_emit),
            ("can receive (direction)", group.direction.can_receive),
            ("confidence >= threshold", confidence >= threshold),
            ("watch only registered", watch_only),
            ("llm rerank", rerank),
        ]
        lines = [f"debug: {describe_group(group)}"]
        for label, ok in checks:
            lines.append(f"  [{'PASS' if ok else 'FAIL'}] {label}")
        lines.append(
            f"  confidence {confidence:.3f} vs threshold {threshold:.3f} "
            f"(k={self.settings.confidence_k:g})"
        )
        lines.append(
            f"  keyword_threshold={self.settings.keyword_threshold:.2f} "
            f"tolerance={self.settings.keyword_tolerance:.2f}"
        )
        lines.append(f"  native={group.native_message_count} chars={group.native_char_count}")
        if llm_ok is not None:
            lines.append(f"  llm available: {llm_ok}")
        if group.intake:
            top = sorted(group.intake.items(), key=lambda kv: kv[1], reverse=True)[:8]
            lines.append("  top taste terms: " + ", ".join(f"{k}:{v:.3f}" for k, v in top))
        else:
            lines.append("  top taste terms: (empty)")

        pending = getattr(self.probe_pipeline, "pending", None)
        if callable(pending):
            lines.append(f"  open windows: {pending()}")

        # ticker 健康状况：它是「抓到了但永远不判定」的唯一嫌疑犯，
        # 所以必须在调试输出里能直接看到，而不是靠猜。
        for attr in ("ticker_alive", "flush_blocked"):
            probe = getattr(self.probe_pipeline, attr, None)
            if callable(probe):
                try:
                    lines.append(f"  {attr.replace('_', ' ')}: {probe()}")
                except Exception:  # pragma: no cover - 探针坏了不该拖垮指令
                    pass
        return CommandResult("\n".join(lines))

    def tags(
        self, event: Any, args: list[str], *, is_admin: bool = False
    ) -> CommandResult:
        """输出一个群的 tag 全貌，或者所有群的 tag 汇总。"""
        if not is_admin:
            return CommandResult("Admin only.", ok=False)

        token = args[0] if args else ""
        if token:
            resolution = resolve(self.storage, event, token)
            if not resolution.ok:
                return CommandResult(f"Target not found: {token}", ok=False)
            group = resolution.group
            assert group is not None
            terms = sorted(group.intake.items(), key=lambda kv: kv[1], reverse=True)
            if not terms:
                return CommandResult(
                    f"{describe_group(group)}\ntags: (none yet - no shit judged here)"
                )
            body = "\n".join(f"  {key}: {value:.4f}" for key, value in terms[:60])
            return CommandResult(
                f"{describe_group(group)}\n"
                f"direction={group.direction.value} confidence={self._confidence(group):.2f}\n"
                f"intake terms ({len(terms)}):\n{body}"
            )

        groups = self.storage.list_groups()
        if not groups:
            return CommandResult("No group registered yet.")
        lines = []
        for group in groups:
            terms = sorted(group.intake.items(), key=lambda kv: kv[1], reverse=True)[:5]
            shown = ", ".join(key for key, _ in terms) or "-"
            lines.append(
                f"{group.group_key} [{group.direction.value}] "
                f"top: {shown}"
            )
        return CommandResult(
            "group tags (top 5 each; /shit tags <target> for full list)\n"
            + "\n".join(lines)
        )

    # ------------------------------------------------------------------ 强制搬运
    def force_delivery(
        self, event: Any, args: list[str], *, is_admin: bool = False, incoming: bool
    ) -> CommandResult:
        """无条件搬运的调试入口。

        incoming=True  : 忽略方向、可信度、相似度，强制“某个源群”把当前这条搬出去
        incoming=False : 忽略方向，强制把当前消息推进“接收侧”的处理链路

        真正执行需要 pipeline 在场，这里只做参数与目标解析，把动作交给 runner。
        """
        if not is_admin:
            return CommandResult("Admin only.", ok=False)

        runner = getattr(self.probe_force, "run", None)
        if not callable(runner):
            return CommandResult(
                "Force delivery is unavailable: pipeline not wired.", ok=False
            )

        token = args[0] if args else ""
        target = resolve(self.storage, event, token) if token else None
        target_key = target.group.group_key if target and target.ok and target.group else ""
        if token and not target_key:
            return CommandResult(f"Target not found: {token}", ok=False)

        try:
            outcome = runner(event, target_key, incoming)
        except Exception as exc:  # pragma: no cover - defensive
            return CommandResult(f"Force delivery failed: {type(exc).__name__}: {exc}", ok=False)
        return CommandResult(outcome if isinstance(outcome, str) else str(outcome))

    # ------------------------------------------------------------------ 目标解析
    def _target(
        self,
        event: Any,
        token: str,
        *,
        is_admin: bool,
        want_unregistered: bool = False,
        require_admin_off_session: bool = False,
    ) -> tuple[GroupProfile | None, CommandResult | None]:
        """解析目标群，并做权限检查。返回 (group, error)。"""
        resolution = resolve(
            self.storage, event, token, want_unregistered=want_unregistered
        )

        if resolution.status == "ambiguous":
            return None, CommandResult(
                render_ambiguous(resolution, self.storage), ok=False
            )
        if resolution.status == "empty":
            return None, CommandResult(
                "No target: run this inside a group or pass a group id.", ok=False
            )
        if resolution.status == "not_found":
            hint = ""
            if not token:
                hint = " This session is not registered; run /shit register."
            return None, CommandResult(
                f"Target not found: {resolution.raw or '(current)'}.{hint}", ok=False
            )

        group = resolution.group
        assert group is not None

        # 操作别的群 = 管理员行为
        off_session = token and group.group_key != current_origin(event)
        if off_session and require_admin_off_session and not is_admin:
            return None, CommandResult(
                "Only an admin can change another group.", ok=False
            )
        return group, None

    def _missing(self, group: GroupProfile, token: str) -> CommandResult:
        return CommandResult(
            f"Group {group.group_key} is not registered "
            f"(/shit register in {token or group.group_key}).",
            ok=False,
        )

    # ------------------------------------------------------------------ 子指令
    def register(
        self, event: Any, args: list[str], *, is_admin: bool = False
    ) -> CommandResult:
        """`/shit register <in|out|both> [target] [threshold]`"""
        direction_token = ""
        target_token = ""
        threshold_token = ""

        for token in args:
            lowered = token.lower()
            if lowered in ("receive", "recv", "emit", "both", "all"):
                direction_token = lowered
                continue
            if _is_threshold(token):
                threshold_token = token
                continue
            if looks_like_target(token) and not target_token:
                target_token = token
                continue
            return CommandResult(
                f"Unexpected argument: {token}\n\n{HELP_TEXT}", ok=False
            )

        direction_token = direction_token or Direction.BOTH.value
        try:
            direction = Direction.parse(direction_token)
        except ValueError as exc:
            return CommandResult(f"{exc} - use receive / emit / both", ok=False)

        threshold = 0.0
        if threshold_token:
            try:
                threshold = self.settings.parse_threshold(threshold_token)
            except ValueError:
                return CommandResult(
                    "Threshold must be 0-1 or a percentage, e.g. 0.8 or 80%", ok=False
                )

        group, error = self._target(
            event,
            target_token,
            is_admin=is_admin,
            want_unregistered=True,
            require_admin_off_session=True,
        )
        if error:
            return error
        assert group is not None

        existing = self.storage.get_group(group.group_key)
        if existing is None:
            group.direction = direction
            group.threshold = threshold
            group.name = group.name or group.group_key
        else:
            existing.direction = direction
            if threshold_token:
                existing.threshold = threshold
            group = existing

        self.storage.upsert_group(group)
        return CommandResult(
            f"[ok] registered {describe_group(group)}\n"
            f"direction : {group.direction.value}\n"
            f"confidence: {self._confidence(group):.2f} "
            f"(threshold {self.settings.threshold_for(group.threshold):.2f})\n"
            f"taste     : {describe_intake(group)}\n"
            f"\nConfidence accrues from the group's own native shit samples."
        )

    def unregister(
        self, event: Any, args: list[str], *, is_admin: bool = False
    ) -> CommandResult:
        token = args[0] if args else ""
        group, error = self._target(
            event, token, is_admin=is_admin, require_admin_off_session=True
        )
        if error:
            return error
        assert group is not None
        if self.storage.delete_group(group.group_key):
            return CommandResult(f"[ok] removed {describe_group(group)}.")
        return self._missing(group, token)

    def toggle(
        self, event: Any, args: list[str], *, enabled: bool, is_admin: bool = False
    ) -> CommandResult:
        """`/shit on <target>` / `/shit off <target>`"""
        token = args[0] if args else ""
        group, error = self._target(
            event, token, is_admin=is_admin, require_admin_off_session=True
        )
        if error:
            return error
        assert group is not None
        stored = self.storage.get_group(group.group_key)
        if stored is None:
            return self._missing(group, token)

        stored.enabled = enabled
        self.storage.upsert_group(stored)
        verb = "enabled (relay on)" if enabled else "paused (silent)"
        return CommandResult(f"[ok] {verb}: {describe_group(stored)}")

    def purge(
        self, event: Any, args: list[str], *, is_admin: bool = False
    ) -> CommandResult:
        if not is_admin:
            return CommandResult("Admin only.", ok=False)
        token = args[0] if args else ""
        group, error = self._target(
            event, token, is_admin=is_admin, require_admin_off_session=True
        )
        if error:
            return error
        assert group is not None
        if self.storage.delete_group(group.group_key):
            return CommandResult(f"[ok] purged {describe_group(group)}.")
        return self._missing(group, token)

    def status(self, event: Any, args: list[str] | None = None) -> CommandResult:
        token = (args or [""])[0]
        resolution = resolve(self.storage, event, token)
        if resolution.status == "ambiguous":
            return CommandResult(render_ambiguous(resolution, self.storage), ok=False)
        if not resolution.ok:
            return CommandResult(
                "Group not registered. Run /shit register both to join.", ok=False
            )

        group = resolution.group
        assert group is not None
        confidence = self._confidence(group)
        threshold = self.settings.threshold_for(group.threshold)
        state = "running" if group.enabled else "paused"
        verdict = "eligible" if confidence >= threshold else "not enough samples"
        return CommandResult(
            f"group     : {describe_group(group)}\n"
            f"direction : {group.direction.value} ({state})\n"
            f"confidence: {confidence:.2f} (threshold {threshold:.2f}) -> {verdict}\n"
            f"native    : {group.native_message_count} shit(s), "
            f"{group.native_char_count} chars\n"
            f"accepted  : {group.accepted_count}  rejected: {group.rejected_count}\n"
            f"taste     : {describe_intake(group)}"
        )

    def list_groups(self) -> CommandResult:
        groups = self.storage.list_groups()
        if not groups:
            return CommandResult("No group registered yet.")
        lines = [f"{len(groups)} group(s):"]
        for index, group in enumerate(groups, start=1):
            state = "" if group.enabled else " [paused]"
            lines.append(
                f"#{index} {group.direction.value:<4} "
                f"conf {self._confidence(group):.2f} "
                f"native {group.native_message_count:<3} "
                f"{group.name or '-'} <{group.group_key}>{state}"
            )
        return CommandResult("\n".join(lines))

    def set_threshold(
        self, event: Any, args: list[str], *, is_admin: bool = False
    ) -> CommandResult:
        """`/shit threshold <0-1|xx%> [target]`"""
        if not args:
            return CommandResult("Usage: /shit threshold <0-1|xx%> [target]", ok=False)

        value_token = args[0]
        token = args[1] if len(args) > 1 else ""
        if not _is_threshold(value_token):
            return CommandResult(
                "Threshold must be 0-1 or a percentage, e.g. 0.8 or 80%", ok=False
            )
        try:
            value = self.settings.parse_threshold(value_token)
        except ValueError:
            return CommandResult("Bad threshold value.", ok=False)

        group, error = self._target(
            event, token, is_admin=is_admin, require_admin_off_session=True
        )
        if error:
            return error
        assert group is not None
        stored = self.storage.get_group(group.group_key)
        if stored is None:
            return self._missing(group, token)

        stored.threshold = value
        self.storage.upsert_group(stored)
        return CommandResult(
            f"[ok] threshold {self.settings.threshold_for(value):.2f} "
            f"for {describe_group(stored)}"
        )

    # ------------------------------------------------------------------ 工具
    def _confidence(self, group: GroupProfile) -> float:
        return confidence_score(
            native_messages=group.native_message_count,
            native_chars=group.native_char_count,
            drift=0.0,
            accepted=group.accepted_count,
            rejected=group.rejected_count,
            k=self.settings.confidence_k,
        )


# ---------------------------------------------------------------------- 工具
def _is_threshold(token: str) -> bool:
    """判断参数是不是阈值。

    群号与阈值都可能是一串数字，靠长度区分：阈值恒在 0-1 或 0-100 之间
    （最多 3 位数 / 带小数点 / 带 %），群号一般是 5 位以上。
    """
    if token.endswith("%"):
        token = token[:-1]
    try:
        float(token)
    except ValueError:
        return False
    return "." in token or (token.isdigit() and len(token) <= 3)


def is_group_event(event: Any) -> bool:
    getter = getattr(event, "get_group_id", None)
    if not callable(getter):
        return False
    return bool(getter())
