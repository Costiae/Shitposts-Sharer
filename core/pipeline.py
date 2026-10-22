"""主流程编排：读到消息 → 判屎 → 更新口味 → 挑目标群 → 搬运 → 记账。

这一层只做“串联”，每个环节都可以单独测试。
"""

from __future__ import annotations

import asyncio
import threading
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .accumulator import ContextAccumulator, Suspicion
from .analyzer import Analyzer
from .bili import BiliResolver, bili_card_hint, extract_bili_ids
from .config import Settings
from .detector import (
    expand_forward,
    is_there_a_shit,
    looks_like_image_only,
    message_from_event,
)
from .llm import LLMClient
from .models import (
    Direction,
    GroupProfile,
    Kind,
    Message,
    ShitRecord,
    normalize_text,
)
from .profile import describe_intake, update_intake
from .router import Router, fingerprint_of
from .sender import build_chain, send_to_group
from .storage import Storage
from .tracelog import Trace, describe_message, describe_segments
from .vector import build_vector, merge_similar_keywords, tokenize


@dataclass(slots=True)
class PipelineStats:
    seen: int = 0
    suspicious: int = 0
    judged: int = 0
    accepted: int = 0
    delivered: int = 0
    failed: int = 0
    rejected_intensity: int = 0
    blocked_safety: int = 0
    last_error: str = ""

    def as_dict(self) -> dict[str, Any]:
        return {
            "seen": self.seen,
            "suspicious": self.suspicious,
            "judged": self.judged,
            "accepted": self.accepted,
            "delivered": self.delivered,
            "failed": self.failed,
            "rejected_intensity": self.rejected_intensity,
            "blocked_safety": self.blocked_safety,
            "last_error": self.last_error,
        }


@dataclass(slots=True)
class Loggerish:
    """尽量用 AstrBot 的 logger，测试里就退化到 print。"""

    inner: Any = None

    def debug(self, text: str) -> None:
        self._log("debug", text)

    def info(self, text: str) -> None:
        self._log("info", text)

    def warning(self, text: str) -> None:
        self._log("warning", text)

    def error(self, text: str) -> None:
        self._log("error", text)

    def _log(self, level: str, text: str) -> None:
        handler = getattr(self.inner, level, None) if self.inner is not None else None
        if callable(handler):
            handler(text)
        else:  # pragma: no cover - 测试环境
            print(f"[{level}] {text}")


@dataclass(slots=True)
class Pipeline:
    context: Any
    settings: Settings
    storage: Storage
    logger: Loggerish = field(default_factory=Loggerish)
    extra: dict[str, Any] = field(default_factory=dict)
    # 下面这些在 __post_init__ 里装配；slots 类不能动态加属性，必须先声明
    llm: Any = field(init=False, repr=False, default=None)
    analyzer: Any = field(init=False, repr=False, default=None)
    router: Any = field(init=False, repr=False, default=None)
    accumulator: Any = field(init=False, repr=False, default=None)
    bili: Any = field(init=False, repr=False, default=None)
    stats: Any = field(init=False, repr=False, default=None)
    _task: Any = field(init=False, repr=False, default=None)
    _flushing: bool = field(init=False, repr=False, default=False)
    _cooldown: Any = field(init=False, repr=False, default=None)
    _tick_seconds: float = field(init=False, repr=False, default=3.0)

    def __post_init__(self) -> None:
        self.llm = LLMClient(self.context)
        self.analyzer = Analyzer(self.settings, self.llm)
        self.router = Router(self.settings)
        self.accumulator = ContextAccumulator(self.settings)
        self.bili = BiliResolver(proxy=str(self.settings.get("proxy") or ""))
        self.stats = PipelineStats()
        self._cooldown = {}
        self._task = None
        self._flushing = False
        self._tick_seconds = 3.0

    # ------------------------------------------------------------------ 生命周期
    async def start(self) -> None:
        if self._task is None or self._task.done():
            self._task = asyncio.create_task(self._ticker())
            self.logger.info("[shit] ticker | started")

    async def stop(self) -> None:
        if self._task and not self._task.done():
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
        self._task = None

    async def _ticker(self) -> None:
        """定时收盘：把攒够时间的现场送去分析。

        这个循环必须**永远活着**。它一死，on_event 照常收消息、照样打
        CAUGHT，但窗口永远不收盘 —— 表现就是「抓到了东西」之后再无下文，
        而且没有任何报错。所以这里的三条规矩：

        1. CancelledError 也要留痕，否则「被谁取消的」永远是谜；
        2. 任何异常都不退出循环，记完日志继续转；
        3. 连续多轮 flush 不出去时主动告警，不让卡死伪装成安静。
        """
        starved = 0
        while True:
            try:
                await asyncio.sleep(self._tick_seconds)
                pending = self.accumulator.pending()
                stuck = self.flush_blocked()
                if stuck:
                    starved += 1
                    self.logger.warning(
                        f"[shit] ticker | flush still running, "
                        f"{pending} window(s) waiting ({starved} tick(s) starved)"
                    )
                else:
                    if starved:
                        self.logger.warning(
                            f"[shit] ticker | recovered after {starved} starved tick(s)"
                        )
                    starved = 0
                    if pending and self.logger is not None:
                        self.logger.debug(
                            f"[shit] ticker | alive, {pending} window(s) pending"
                        )
                await self.flush_due()
            except asyncio.CancelledError:
                self.logger.warning("[shit] ticker | CANCELLED, loop is ending")
                raise
            except Exception as exc:
                self.stats.last_error = f"{type(exc).__name__}: {exc}"
                self.logger.error(
                    f"[shit] ticker | CRASHED: {type(exc).__name__}: {exc}"
                )
                # 关键：不 raise，让循环活下去。否则一次 closed database
                # 就永久废掉整个收盘机制。
                try:
                    await asyncio.sleep(self._tick_seconds)
                except asyncio.CancelledError:
                    self.logger.warning("[shit] ticker | CANCELLED, loop is ending")
                    raise

    def flush_blocked(self) -> bool:
        """上一轮 flush 是否还没回来（调试与心跳都用这个）。"""
        return self._flushing

    async def flush_due(self, now: float | None = None) -> list[Any]:
        if self._flushing:
            # 不要静默返回：上一轮还卡着的话，这里每 3 秒无声跳过，
            # 现场看起来就跟「ticker 死了」一模一样。
            return []
        self._flushing = True
        started = time.time()
        try:
            results = []
            for group_key, suspicion in self.accumulator.due(now):
                results.append(await self._handle_with_timeout(suspicion))
            return results
        finally:
            self._flushing = False
            spent = time.time() - started
            if spent > self._tick_seconds * 2:
                self.logger.warning(
                    f"[shit] ticker | flush took {spent:.1f}s (blocked the loop)"
                )

    async def _handle_with_timeout(self, suspicion: Suspicion) -> Any:
        """单条判定加超时。

        LLM 的 429 重试可能长达数分钟（retry in 59s × 5），而
        handle_suspicion 里没有任何超时保护 —— 一条卡住就拖死整个
        收盘循环。这里给它兜底，超时后把现场标记为失败并继续。
        """
        limit = float(self.settings.get("judge_timeout_sec") or 180)
        try:
            return await asyncio.wait_for(
                self.handle_suspicion(suspicion), timeout=limit
            )
        except asyncio.TimeoutError:
            self.stats.last_error = f"judge timed out after {limit:.0f}s"
            self.logger.error(
                f"[shit] judge | TIMEOUT after {limit:.0f}s "
                f"({suspicion.message.group_key}) - likely LLM retry storm"
            )
            return None
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            # 单个现场炸了不该拖累别的群
            self.stats.last_error = f"{type(exc).__name__}: {exc}"
            self.logger.error(
                f"[shit] judge | FAILED: {type(exc).__name__}: {exc} "
                f"({suspicion.message.group_key})"
            )
            return None

    # ------------------------------------------------------------------ 入口
    async def on_event(self, event: Any) -> None:
        """每条群消息都会进来。"""
        trace = Trace(self.logger, str(self.settings.get("log_level") or "normal"))

        if not self.settings.get("enabled"):
            trace.gate("global")
            trace.skip("plugin disabled")
            return

        group_key = str(getattr(event, "unified_msg_origin", "") or "")
        if not group_key:
            trace.gate("origin")
            trace.skip("no unified_msg_origin")
            return

        trace.gate("group")
        group = self.storage.get_group(group_key)
        if self.settings.watch_only_registered and group is None:
            trace.skip(f"not registered ({group_key})")
            return
        if group is not None and not group.enabled:
            trace.skip(f"group paused ({group_key})")
            return

        trace.gate("decode")
        message = message_from_event(event, group_key)
        trace.step(f"decoded {describe_segments(message)}")

        if self.settings.get("expand_forward", True):
            before = sum(1 for seg in message.segments if seg.kind == "forward" and not seg.children)
            message = await expand_forward(message, self.context)
            after = sum(1 for seg in message.segments if seg.kind == "forward" and not seg.children)
            if before:
                trace.step(f"forward expand: {before} hollow -> {after} hollow")
                if after and not before - after:
                    trace.step("forward expand FAILED (no nodes returned)")

        message = self._enrich(message)
        self.accumulator.observe(message, has_bili=bool(self._bili_refs(message.text())))

        if group is not None and not group.direction.can_emit:
            # receive 群：它自己说的话不往外搬，但消息仍进上下文（作为语境）
            trace.skip(f"group is receive-only, message kept as context only")
            return

        self.stats.seen += 1
        trace.gate("detect")
        interesting, reason = is_there_a_shit(message, self.settings)
        if not interesting:
            soft = self.settings.soft_text_min_chars if self.settings.loose_detect else 0
            trace.skip(
                f"not suspicious need={self.settings.text_min_chars}chars "
                f"soft={soft or '-'} got={message.char_count()} "
                f"kinds={sorted(message.kinds())}"
            )
            return

        now = time.time()
        last = self._cooldown.get(group_key, 0.0)
        cooldown = float(self.settings.get("trigger_cooldown_sec") or 0)
        if now - last < cooldown:
            trace.skip(f"cooldown {now - last:.1f}s < {cooldown:.0f}s")
            return
        self._cooldown[group_key] = now

        self.stats.suspicious += 1
        trace.caught(f"{reason} | {describe_message(message)}")
        self.accumulator.open(message, reason, trace=trace)

    def _bili_refs(self, text: str):
        return extract_bili_ids(text)

    def _enrich(self, message: Message) -> Message:
        """把卡片消息里的可读信息塞进文本，让 LLM 有的看。"""
        for seg in message.segments:
            if seg.kind in ("json", "miniapp", "xml") and seg.text:
                hint = bili_card_hint(seg.text)
                if hint:
                    seg.text = hint
        return message

    # ------------------------------------------------------------------ 判屎
    async def handle_suspicion(self, suspicion: Suspicion) -> dict[str, Any] | None:
        group_key = suspicion.message.group_key
        group = self.storage.get_group(group_key)
        trace = suspicion.trace or Trace(self.logger, str(self.settings.get("log_level") or "normal"))
        trace.gate("judge")

        if group is None and self.settings.watch_only_registered:
            trace.skip("not registered at judge time")
            return None

        self.stats.judged += 1
        message = suspicion.message
        text = self._enrich(message).text()
        trace.step(f"window closed after {suspicion.age():.1f}s, "
                   f"context={len(suspicion.context_before)} msgs, "
                   f"extra={len(suspicion.extra_messages)}")

        bili_infos = []
        if self.settings.get("bili_info") and self._bili_refs(text).any():
            bili_infos = await self.bili.resolve(text)
            trace.step(f"bili resolved {len(bili_infos)} item(s)")

        kind = message.kind_hint()
        trace.step(f"prompt input: kind={kind} chars={len(text)} "
                   f"images={sum(1 for s in message.segments if s.kind == 'image')}")
        if trace.level == "full":
            trace.step(f"prompt content: {describe_message(message, limit=600)}")

        verdict = await self.analyzer.judge(
            suspicion,
            group.name if group else group_key,
            bili_infos=bili_infos,
            kind=kind,
        )

        if not verdict.ok:
            self.stats.last_error = "LLM 未返回可解析结果"
            trace.error(
                f"judge FAILED: raw response unparseable "
                f"(raw={str(verdict.raw)[:300]!r})"
            )
            return None

        analysis = verdict.analysis

        # 安全闸门：涉及未成年人的性化内容永不搬运，也不入库。
        # 这是硬性规则，不提供配置开关。
        if analysis.minor_sexual:
            self.stats.blocked_safety += 1
            trace.skip("safety: sexualized minor content is never relayed")
            return None

        # 留痕：判没判、判成啥、分数够不够，一眼可辨。
        # 没有这行时，“判成不是屎”和“分数被阈值挡掉”在日志里长得一模一样。
        min_intensity = float(self.settings.get("shit_min_intensity") or 0.0)
        trace.step(
            f"model raw: is_shit={analysis.is_shit} intensity={analysis.intensity:.2f} "
            f"tags={analysis.tags} keywords={analysis.keywords[:5]} "
            f"image_tags={analysis.image_tags[:5]} nsfw={analysis.nsfw}"
        )

        if not analysis.is_shit:
            trace.skip(
                f"model says NOT shit (intensity={analysis.intensity:.2f} "
                f"reason={analysis.reason!r})"
            )
            return None

        if analysis.intensity < min_intensity:
            self.stats.rejected_intensity += 1
            trace.skip(
                f"intensity {analysis.intensity:.2f} < min {min_intensity:.2f} "
                f"(reason={analysis.reason!r})"
            )
            return None

        self.stats.accepted += 1
        trace.caught(
            f"is_shit=true intensity={analysis.intensity:.2f} "
            f"reason={analysis.reason!r}"
        )

        merge_threshold = float(self.settings.get("keyword_merge_threshold") or 0.6)
        keywords = merge_similar_keywords(
            [*analysis.keywords, *analysis.context_keywords],
            threshold=merge_threshold,
        )

        # 图片标签：角色 / IP / 属性 / 风格，和文字关键词一样参与口味匹配。
        # 只有图片消息才打；无图或关闭 image_tagging 时为空，退回纯文字匹配。
        image_tags: list[str] = []
        if self.settings.image_tagging and kind == "image":
            image_tags = merge_similar_keywords(
                [*analysis.image_tags, *analysis.characters, *analysis.copyrights],
                threshold=merge_threshold,
            )
            if analysis.nsfw:
                image_tags.append("NSFW")
            if image_tags:
                trace.step(f"image tags: {image_tags} nsfw={analysis.nsfw}")

        image_vector = (
            build_vector(
                image_tags,
                "",
                analysis.intensity,
                merge_threshold=merge_threshold,
            )
            if image_tags
            else {}
        )

        blob = normalize_text(
            "\n".join(
                part
                for part in [
                    text,
                    " ".join(info.title for info in bili_infos if info.title),
                    " ".join(analysis.tags),
                    " ".join(image_tags),
                ]
                if part
            )
        )
        vector = build_vector(
            [*keywords, *image_tags],
            blob,
            analysis.intensity,
            analysis.tags,
            merge_threshold=merge_threshold,
        )
        if not vector:
            # 兜底：至少用正文分词，不然这条屎永远匹配不上任何群
            vector = build_vector(
                [],
                blob,
                analysis.intensity,
                analysis.tags,
                merge_threshold=merge_threshold,
            )

        record = ShitRecord(
            shit_id=uuid.uuid4().hex,
            source_group=group_key,
            ts=time.time(),
            kind=Kind(analysis_kind(kind)),
            text_blob=blob,
            keywords=keywords,
            vector=vector,
            tags=list(analysis.tags),
            image_tags=image_tags,
            image_vector=image_vector,
            nsfw=analysis.nsfw,
            intensity=analysis.intensity,
            context_included=self.settings.get("with_context", True),
            payload=[seg.to_dict() for seg in message.segments],
            context_text=self._compose_context(suspicion),
            reason=analysis.reason,
        )
        record.fingerprint = fingerprint_of(record)

        # 去重：别人已经先把同一坨屎搬过了
        existing = self.storage.find_by_fingerprint(record.fingerprint)
        if existing is not None:
            record.incoming = True
            trace.skip(
                f"duplicate (first seen in {existing.source_group}, "
                f"fingerprint={record.fingerprint[:12]})"
            )
            if group is not None:
                self.storage.upsert_group(group)
            self.storage.add_shit(record)
            return {"duplicate": True, "shit_id": existing.shit_id}

        self.storage.add_shit(record)

        # 更新来源群的口味（外来内容只影响口味，不增加原生可信度）
        if group is not None:
            update_intake(group, record, self.settings, is_native=not record.incoming)
            self.storage.upsert_group(group)

        trace.gate("route")
        plan = self.router.keyword_plan(
            record,
            self.storage.list_groups(Direction.RECEIVE),
            exclude={group_key},
        )
        trace.step(
            f"keyword candidates={len(plan.candidates)} rejected={len(plan.rejected)}"
        )
        for key, why in plan.rejected[:10]:
            trace.step(f"  rejected {key}: {why}")

        if not plan.candidates:
            trace.skip("no candidate group passed the keyword+confidence gate")

        # 第二层：LLM 复核候选群（开关由 analysis_provider_id / 提示词控制）
        if plan.candidates and self.settings.get("llm_rerank", True) and self.llm.available:
            top = plan.candidates[: int(self.settings.get("llm_candidates") or 3)]
            groups = {
                group.group_key: group for group in self.storage.list_groups(Direction.RECEIVE)
            }
            scores = await self.analyzer.score_groups(
                shit_text=blob[:1500],
                shit_keywords=[*keywords, *image_tags],
                groups=[groups[item.group_key] for item in top if item.group_key in groups],
                umo=record.source_group,
            )
            if scores:
                plan = self.router.apply_llm_scores(plan, scores, list(groups.values()))
                trace.step(f"llm rerank scores: {scores}")
            else:
                trace.step("llm rerank returned nothing, keeping keyword order")

        delivered = []
        for candidate in plan.candidates:
            target = self.storage.get_group(candidate.group_key)
            if target is None or not self.router.can_receive(target):
                trace.step(f"skip {candidate.group_key}: not receivable")
                continue
            if not self.storage.record_delivery(
                record.shit_id,
                target.group_key,
                candidate.score,
                candidate.confidence,
                candidate.source,
                "pending",
            ):
                trace.step(f"skip {target.group_key}: already delivered")
                continue  # 已经搬过

            if self.settings.get("dry_run"):
                self.storage.mark_delivery(record.shit_id, target.group_key, "dry_run")
                delivered.append(target.group_key)
                trace.result(
                    f"DRY RUN would deliver to {target.group_key} "
                    f"(score={candidate.score:.2f} confidence={candidate.confidence:.2f})"
                )
                continue

            ok = await self._deliver(record, target, group, trace=trace)
            self.storage.mark_delivery(
                record.shit_id, target.group_key, "ok" if ok else "fail"
            )
            if ok:
                self.stats.delivered += 1
                delivered.append(target.group_key)
                target.accepted_count += 1
                self.storage.upsert_group(target)
                trace.result(
                    f"DELIVERED to {target.group_key} "
                    f"(score={candidate.score:.2f} confidence={candidate.confidence:.2f})"
                )
            else:
                self.stats.failed += 1
                trace.warn(f"delivery to {target.group_key} FAILED")

        trace.gate("result")
        trace.result(
            f"shit={record.shit_id[:8]} from={group_key} "
            f"intensity={analysis.intensity:.2f} keywords={keywords[:5]} "
            f"image_tags={image_tags[:5] or 'none'} delivered={delivered or 'none'}"
        )
        if trace.level == "full":
            trace.step(f"trail: {trace.summary()}")
        return {
            "shit_id": record.shit_id,
            "keywords": keywords,
            "delivered": delivered,
            "rejected": plan.rejected,
        }

    def _compose_context(self, suspicion: Suspicion) -> str:
        if not self.settings.get("with_context", True):
            return ""
        limit = int(self.settings.get("max_context_messages") or 6)
        before = self.accumulator.render_context(suspicion, limit=limit)
        extra = self.accumulator.render_extra(suspicion)
        parts = [part for part in (before, extra) if part]
        return "\n".join(parts)

    async def _deliver(
        self,
        record: ShitRecord,
        target: GroupProfile,
        source_group: GroupProfile | None,
        *,
        trace: Trace | None = None,
    ) -> bool:
        chain = build_chain(
            record,
            label=str(self.settings.get("source_label") or "【搬屎】"),
            source_name=(
                ""
                if self.settings.get("anon_source", True)
                else (source_group.name if source_group else record.source_group)
            ),
            with_context=bool(self.settings.get("with_context", True)),
        )
        if trace is not None and trace.level == "full":
            trace.step(f"built chain with {len(chain)} component(s)")
        result = await send_to_group(self.context, target.group_key, chain)
        if not result.ok:
            self.stats.last_error = result.detail
            if trace is not None:
                trace.warn(f"send to {target.group_key} failed: {result.detail}")
            else:
                self.logger.warning(
                    f"[shit] deliver | failed to {target.group_key}: {result.detail}"
                )
        return result.ok

    # ------------------------------------------------------------------ 调试
    def pending(self) -> int:
        """当前有多少个上下文窗口还在积累（调试指令用）。"""
        return self.accumulator.pending()

    def ticker_alive(self) -> bool:
        """收盘循环是否还活着（调试指令用）。"""
        return self._task is not None and not self._task.done()

    def _message_from_event(self, event: Any):
        group_key = str(getattr(event, "unified_msg_origin", "") or "")
        if not group_key:
            return None
        return self._enrich(message_from_event(event, group_key))

    async def _message_from_event_expanded(self, event: Any):
        """同 _message_from_event，但把合并转发展开（force 指令用）。"""
        message = self._message_from_event(event)
        if message is None:
            return None
        if self.settings.get("expand_forward", True):
            message = await expand_forward(message, self.context)
        return message

    async def force_delivery(
        self,
        event: Any,
        target_key: str = "",
        incoming: bool = True,
    ) -> dict[str, Any]:
        """无条件搬运：跳过方向、可信度、相似度与 LLM 复核。

        ``incoming=True``  当前消息当作“源屎”，直接推到所有能收的群，
                           ``target_key`` 非空时只推到那个群；
        ``incoming=False`` 当前消息当作“被搬入的屎”，只更新目标群口味，
                           不往外发。

        这是调试入口，会真实发消息，所以只挂在管理员指令下。
        """
        message = await self._message_from_event_expanded(event)
        if message is None:
            return {"ok": False, "detail": "no session"}

        text = message.text()
        group = self.storage.get_group(message.group_key)
        blob = normalize_text(text) or f"{message.kind_hint()}"
        record = ShitRecord(
            shit_id=f"force-{uuid.uuid4().hex[:12]}",
            source_group=message.group_key,
            ts=time.time(),
            kind=Kind(analysis_kind(message.kind_hint())),
            text_blob=blob,
            keywords=[token for token in tokenize(blob)[:8]],
            vector=build_vector([], blob, 0.5),
            payload=[seg.to_dict() for seg in message.segments],
            reason="forced (debug)",
        )
        record.fingerprint = fingerprint_of(record)
        self.storage.add_shit(record)

        if not incoming:
            target = self.storage.get_group(target_key) if target_key else group
            if target is None:
                return {"ok": False, "detail": f"target not registered: {target_key}"}
            update_intake(target, record, self.settings, is_native=False)
            self.storage.upsert_group(target)
            return {"ok": True, "detail": f"intake updated for {target.group_key}"}

        groups = self.storage.list_groups()
        if target_key:
            groups = [item for item in groups if item.group_key == target_key]

        delivered: list[str] = []
        failed: list[str] = []
        for item in groups:
            if not item.enabled:
                continue
            ok = await self._deliver(record, item, group)
            self.storage.record_delivery(
                record.shit_id, item.group_key, 1.0, 1.0, "forced", "ok" if ok else "fail"
            )
            if ok:
                delivered.append(item.group_key)
                self.stats.delivered += 1
            else:
                failed.append(item.group_key)
                self.stats.failed += 1

        return {
            "ok": bool(delivered),
            "detail": f"delivered={delivered or '-'} failed={failed or '-'}",
        }

    def run(self, event: Any, target_key: str, incoming: bool) -> str:
        """同步桥：指令层是同步的，这里把协程跑完并返回文本。"""
        try:
            result = asyncio.run(
                self.force_delivery(event, target_key, incoming)
            )
        except RuntimeError:
            # 已经有事件循环在跑（AstrBot 主循环）：开新线程跑
            box: dict[str, Any] = {}

            def _worker() -> None:
                box["value"] = asyncio.run(
                    self.force_delivery(event, target_key, incoming)
                )

            thread = threading.Thread(target=_worker)
            thread.start()
            thread.join()
            result = box.get("value", {"ok": False, "detail": "worker failed"})
        except Exception as exc:  # pragma: no cover - defensive
            return f"force failed: {type(exc).__name__}: {exc}"

        return str(result.get("detail", result))

    # ------------------------------------------------------------------ 统计
    def describe_group(self, group: GroupProfile) -> str:
        recent = self.storage.recent_deliveries(group.group_key, limit=5)
        confidence = group.confidence(self.settings.confidence_k)
        lines = [
            f"群：{group.name or group.group_key}",
            f"方向：{group.direction.label()}（{group.direction.describe()}）",
            f"可信度：{confidence:.2f}"
            f"（原生屎 {group.native_message_count} 条 / {group.native_char_count} 字，"
            f"偏移惩罚 {group.rejected_count} 次）",
            f"口味：{describe_intake(group)}",
        ]
        if recent:
            lines.append("最近搬运：")
            for row in recent:
                lines.append(
                    f"  - {row['shit_id'][:8]} 分数 {row['score']:.2f} "
                    f"可信度 {row['confidence']:.2f} [{row['status']}]"
                )
        return "\n".join(lines)


def analysis_kind(kind: str) -> str:
    mapping = {
        "forward": Kind.FORWARD.value,
        "image": Kind.IMAGE.value,
        "text": Kind.TEXT.value,
    }
    return mapping.get(kind, Kind.MIXED.value)


def ensure_data_dir(root: Path) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    return root
