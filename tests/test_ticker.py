"""Ticker robustness tests.

The failure this guards against: on_event keeps logging CAUGHT, but the
90s window never closes, so no judgment ever happens and nothing in the
log says why.

Three ways that used to happen silently:
  1. an exception killed the ticker loop permanently;
  2. a long LLM call (429 retry storm) held the flush lock forever while
     the ticker span in circles;
  3. cancellation exited the loop without a trace.

Each test below fails on the old implementation.
"""

from __future__ import annotations

import asyncio
import sys
import time
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.config import DEFAULTS, Settings  # noqa: E402
from core.models import Direction, GroupProfile  # noqa: E402
from core.pipeline import Pipeline  # noqa: E402
from core.storage import Storage  # noqa: E402
from tests.test_core import scratch_path  # noqa: E402
from tests.test_tracelog import CapturingLogger, FakeContext, FakeEvent  # noqa: E402

LONG = "这是一段足够长的群聊内容" * 6

_PATCHED: list[tuple] = []


def _patch(obj, name, func):
    """Pipeline 是 slots dataclass，实例属性只读，只能打在类上。

    返回后由 _unpatch_all 恢复，避免测试之间互相污染。
    """
    cls = type(obj)
    _PATCHED.append((cls, name, cls.__dict__[name]))
    setattr(cls, name, func)


def _unpatch_all():
    while _PATCHED:
        cls, name, original = _PATCHED.pop()
        setattr(cls, name, original)




class TestTickerSurvives(unittest.TestCase):
    def setUp(self):
        import core.sender as sender_module
        from tests.test_pipeline import _stub_plain

        self._orig = sender_module._plain
        sender_module._plain = _stub_plain
        self.db = scratch_path("ticker")
        self.storage = Storage(self.db)
        self.storage.upsert_group(
            GroupProfile(
                group_key="aiocqhttp:GroupMessage:1",
                name="g",
                direction=Direction.EMIT,
            )
        )

    def tearDown(self):
        import core.sender as sender_module

        sender_module._plain = self._orig
        _unpatch_all()
        self.storage.close()
        for suffix in ("", "-wal", "-shm"):
            Path(str(self.db) + suffix).unlink(missing_ok=True)

    def _pipeline(self, **overrides):
        raw = dict(DEFAULTS)
        raw.update(trigger_cooldown_sec=0)
        raw.update(overrides)
        logger = CapturingLogger()
        pipe = Pipeline(
            context=FakeContext('{"is_shit": false}'),
            settings=Settings(raw),
            storage=self.storage,
            logger=logger,
        )
        pipe._tick_seconds = 0.01
        return pipe, logger

    # ---------------------------------------------------------- 1. 异常不致死
    def test_ticker_survives_exception(self):
        """一次异常之后，循环必须继续转（旧实现会永久退出）。"""
        pipe, logger = self._pipeline()
        calls = {"n": 0}

        async def boom(_self):
            calls["n"] += 1
            if calls["n"] == 1:
                raise RuntimeError("Cannot operate on a closed database.")
            return []

        _patch(pipe, "flush_due", boom)

        async def run():
            task = asyncio.create_task(pipe._ticker())
            await asyncio.sleep(0.15)
            alive = not task.done()
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass
            return alive

        alive = asyncio.run(run())
        # 旧实现在第一次异常时就退出循环，calls["n"] 会停在 1
        self.assertGreaterEqual(calls["n"], 2, "ticker stopped after the exception")
        # 循环还活着（是在测试里被 cancel 掉的，不是自己死的）
        self.assertTrue(alive, "ticker exited on its own after the exception")
        self.assertIn("CRASHED", logger.text())
        self.assertIn("closed database", logger.text())

    def test_exception_is_recorded_in_last_error(self):
        pipe, _ = self._pipeline()

        async def boom(_self):
            raise RuntimeError("closed database")

        _patch(pipe, "flush_due", boom)

        async def run():
            task = asyncio.create_task(pipe._ticker())
            await asyncio.sleep(0.05)
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass

        asyncio.run(run())
        self.assertIn("closed database", str(pipe.stats.last_error))

    # ---------------------------------------------------------- 2. 取消也留痕
    def test_cancellation_is_logged(self):
        pipe, logger = self._pipeline()

        async def empty(_self):
            return []

        _patch(pipe, "flush_due", empty)

        async def run():
            task = asyncio.create_task(pipe._ticker())
            await asyncio.sleep(0.03)
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass

        asyncio.run(run())
        self.assertIn("CANCELLED", logger.text())

    # ---------------------------------------------------------- 3. 卡住要告警
    def test_blocked_flush_is_logged_not_silent(self):
        """上一轮 flush 没回来时，每轮都要留痕，不能静默空转。"""
        pipe, logger = self._pipeline()
        pipe._flushing = True  # 模拟上一轮卡在 LLM 调用里

        async def run():
            task = asyncio.create_task(pipe._ticker())
            await asyncio.sleep(0.15)
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass

        asyncio.run(run())
        self.assertIn("still running", logger.text())
        self.assertIn("starved", logger.text())

    def test_blocked_flag_reported(self):
        pipe, _ = self._pipeline()
        self.assertFalse(pipe.flush_blocked())
        pipe._flushing = True
        self.assertTrue(pipe.flush_blocked())

    def test_recovery_is_logged(self):
        pipe, logger = self._pipeline()
        pipe._flushing = True
        state = {"n": 0}

        async def run():
            task = asyncio.create_task(pipe._ticker())
            await asyncio.sleep(0.08)
            pipe._flushing = False  # 卡住的 flush 终于回来了
            await asyncio.sleep(0.08)
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass

        asyncio.run(run())
        self.assertIn("recovered", logger.text())

    # ---------------------------------------------------------- 4. 单条超时
    def test_single_suspicion_timeout_does_not_drag_the_loop(self):
        """429 重试导致的单条卡死必须被 wait_for 切断。"""
        import core.accumulator as acc

        pipe, logger = self._pipeline(judge_timeout_sec=0.05)

        async def hang(_self, suspicion):
            await asyncio.sleep(10)
            return None

        _patch(pipe, "handle_suspicion", hang)
        suspicion = acc.Suspicion(
            message=__import__("core.models", fromlist=["Message"]).Message(
                group_key="aiocqhttp:GroupMessage:1", segments=[]
            ),
            reason="test",
        )

        started = time.time()
        result = asyncio.run(pipe._handle_with_timeout(suspicion))
        spent = time.time() - started

        self.assertIsNone(result)
        self.assertLess(spent, 2.0, "timeout did not cut the call")
        self.assertIn("TIMEOUT", logger.text())
        self.assertIn("timed out", str(pipe.stats.last_error))

    def test_single_suspicion_crash_does_not_kill_others(self):
        pipe, logger = self._pipeline()
        import core.accumulator as acc
        from core.models import Message

        async def boom(_self, suspicion):
            raise RuntimeError("one group broke")

        _patch(pipe, "handle_suspicion", boom)
        suspicion = acc.Suspicion(
            message=Message(group_key="aiocqhttp:GroupMessage:1", segments=[]),
            reason="test",
        )
        result = asyncio.run(pipe._handle_with_timeout(suspicion))
        self.assertIsNone(result)
        self.assertIn("FAILED", logger.text())
        self.assertIn("one group broke", logger.text())

    # ---------------------------------------------------------- 5. 自检探针
    def test_ticker_alive_probe(self):
        pipe, _ = self._pipeline()
        self.assertFalse(pipe.ticker_alive())

        async def run():
            await pipe.start()
            alive = pipe.ticker_alive()
            await pipe.stop()
            return alive

        self.assertTrue(asyncio.run(run()))
        self.assertFalse(pipe.ticker_alive())

    def test_start_is_logged(self):
        pipe, logger = self._pipeline()

        async def run():
            await pipe.start()
            await pipe.stop()

        asyncio.run(run())
        self.assertIn("ticker | started", logger.text())


class TestFlushSlowWarning(unittest.TestCase):
    def setUp(self):
        self.db = scratch_path("slowflush")
        self.storage = Storage(self.db)

    def tearDown(self):
        _unpatch_all()
        self.storage.close()
        for suffix in ("", "-wal", "-shm"):
            Path(str(self.db) + suffix).unlink(missing_ok=True)

    def test_slow_flush_warns(self):
        raw = dict(DEFAULTS)
        logger = CapturingLogger()
        pipe = Pipeline(
            context=FakeContext("{}"),
            settings=Settings(raw),
            storage=self.storage,
            logger=logger,
        )
        pipe._tick_seconds = 0.001

        import core.accumulator as acc
        from core.models import Message

        async def slow(_self, suspicion):
            await asyncio.sleep(0.05)
            return None

        _patch(pipe, "handle_suspicion", slow)
        pipe.accumulator._open["g"] = acc.Suspicion(
            message=Message(group_key="g", segments=[]), reason="t"
        )
        asyncio.run(pipe.flush_due(now=time.time() + 9999))
        self.assertIn("blocked the loop", logger.text())


if __name__ == "__main__":
    unittest.main()
