"""Lifecycle-owned, single-flight API projector. SQL never runs on the event loop."""

from __future__ import annotations

import asyncio
import time
from concurrent.futures import ThreadPoolExecutor

from activekg.candidate_history.contracts import HistoryConfig
from activekg.candidate_history.repository import HistoryRepository, HistoryUnavailable


class HistoryProjector:
    def __init__(self, repository: HistoryRepository, config: HistoryConfig):
        self._repository = repository
        self._config = config
        self._executor: ThreadPoolExecutor | None = None
        self._task: asyncio.Task | None = None
        self._stop = asyncio.Event()
        self._last_success: float | None = None
        self._last_error: str | None = None

    def start(self) -> None:
        if not self._config.enabled or self._task is not None:
            return
        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="candidate-history")
        self._stop.clear()
        self._task = asyncio.create_task(self._run())

    async def _run(self) -> None:
        assert self._executor is not None
        while not self._stop.is_set():
            future = asyncio.get_running_loop().run_in_executor(
                self._executor,
                self._repository.step,
                self._config.new_limit,
                self._config.retry_limit,
            )
            try:
                # Shield prevents task cancellation from orphaning the DB turn.
                await asyncio.shield(future)
                self._last_success = time.monotonic()
                self._last_error = None
            except asyncio.CancelledError:
                try:
                    await asyncio.shield(future)
                except Exception:
                    pass
                raise
            except HistoryUnavailable:
                self._last_error = "temporarily_unavailable"
            except Exception:
                self._last_error = "temporarily_unavailable"
            if not self._stop.is_set():
                try:
                    await asyncio.wait_for(self._stop.wait(), self._config.poll_ms / 1000)
                except TimeoutError:
                    pass

    def healthy(self) -> bool:
        if not self._config.enabled:
            return True
        age = None if self._last_success is None else time.monotonic() - self._last_success
        return bool(
            self._task
            and not self._task.done()
            and self._last_error is None
            and age is not None
            and 0 <= age <= 30
        )

    async def stop(self) -> None:
        self._stop.set()
        try:
            if self._task is not None:
                # Statement, lock, idle and connect timeouts bound the current turn.
                # Do not begin another turn while the worker is draining it.
                await self._task
        finally:
            self._task = None
            if self._executor is not None:
                self._executor.shutdown(wait=True, cancel_futures=True)
                self._executor = None
