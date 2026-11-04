"""CPU worker pools shared by tuning, inference and evaluation (2D and 3D).

The GPU forward pass stays in the main process; everything around it
(NIfTI loading, reorientation / resampling, peak extraction, decoding, PNG
and NIfTI writing, metric computation) runs in a pool of worker processes
so it overlaps with the GPU work instead of serialising with it.

Workers are started with the ``spawn`` method: the main process usually has
CUDA initialised already, and forking a CUDA process is unsafe. Every
function sent to a pool must therefore be defined at module level.

``num_workers = 0`` disables the pool and runs everything inline, which is
handy for debugging (tracebacks stay in-process).
"""

from __future__ import annotations

import collections
import multiprocessing as mp
import os
import sys
from concurrent.futures import Executor, Future, ProcessPoolExecutor
from typing import Any, Callable, Deque, Iterable, Iterator, List, Optional

DEFAULT_MAX_WORKERS = 8

# Per-worker thread caps: N workers each spinning up an all-core OpenMP/BLAS
# pool would oversubscribe the machine.
_THREAD_ENV_VARS = (
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
)


def available_cpus() -> int:
    """CPUs this process may run on (honours SLURM / cgroup affinity masks)."""
    if hasattr(os, "sched_getaffinity"):
        try:
            return max(1, len(os.sched_getaffinity(0)))
        except OSError:
            pass
    return max(1, os.cpu_count() or 1)


def resolve_num_workers(value: Any = "auto", cap: int = DEFAULT_MAX_WORKERS) -> int:
    """``"auto"`` / ``None`` → ``min(available CPUs - 1, cap)``; ints are honoured as-is.

    One CPU is left for the main process, which drives the GPU.
    """
    if isinstance(value, str) and value.strip().lstrip("-").isdigit():
        value = int(value)
    if isinstance(value, int) and not isinstance(value, bool) and value >= 0:
        return value
    return max(0, min(available_cpus() - 1, cap))


def _worker_init(extra_init: Optional[Callable], extra_args: tuple) -> None:
    for var in _THREAD_ENV_VARS:
        os.environ[var] = "1"
    # The console-script wrapper is re-imported in every spawned worker, so
    # torch may already be loaded here; keep its CPU pool to a single thread.
    torch = sys.modules.get("torch")
    if torch is not None:
        try:
            torch.set_num_threads(1)
        except Exception:
            pass
    if extra_init is not None:
        extra_init(*extra_args)


def make_executor(
    num_workers: int,
    initializer: Optional[Callable] = None,
    initargs: tuple = (),
) -> Optional[ProcessPoolExecutor]:
    """Spawn-based process pool, or ``None`` when ``num_workers <= 0``."""
    if num_workers <= 0:
        if initializer is not None:
            initializer(*initargs)
        return None
    return ProcessPoolExecutor(
        max_workers=num_workers,
        mp_context=mp.get_context("spawn"),
        initializer=_worker_init,
        initargs=(initializer, initargs),
    )


class _InlineFuture:
    """Already-resolved stand-in for ``Future`` used when there is no pool."""

    def __init__(self, fn: Callable, args: tuple):
        self._value = fn(*args)

    def result(self):
        return self._value


def submit(executor: Optional[Executor], fn: Callable, *args):
    """``executor.submit`` that falls back to an inline call without a pool."""
    if executor is None:
        return _InlineFuture(fn, args)
    return executor.submit(fn, *args)


def imap_ordered(
    executor: Optional[Executor],
    fn: Callable,
    items: Iterable,
    lookahead: int,
) -> Iterator:
    """Ordered ``map`` that keeps at most ``lookahead`` jobs in flight.

    Unlike ``Executor.map`` it does not submit every item up front, so large
    results (e.g. preprocessed 3D volumes) never pile up in memory faster
    than the consumer drains them.
    """
    if executor is None:
        for item in items:
            yield fn(item)
        return
    lookahead = max(1, int(lookahead))
    pending: Deque[Future] = collections.deque()
    it = iter(items)
    try:
        for item in it:
            pending.append(executor.submit(fn, item))
            if len(pending) >= lookahead:
                break
        while pending:
            result = pending.popleft().result()
            for item in it:
                pending.append(executor.submit(fn, item))
                break
            yield result
    finally:
        for f in pending:
            f.cancel()


class BoundedSubmitter:
    """Fire-and-collect background jobs with a cap on how many are queued.

    ``submit`` blocks (waiting on the oldest job) once ``max_pending`` jobs
    are in flight, which bounds the memory held by queued arguments.
    ``drain`` waits for everything and returns the results in submit order;
    worker exceptions are re-raised there (or earlier, while blocking).
    """

    def __init__(self, executor: Optional[Executor], max_pending: int):
        self._executor = executor
        self._max_pending = max(1, int(max_pending))
        self._pending: Deque = collections.deque()
        self._done: List[Any] = []

    def submit(self, fn: Callable, *args) -> None:
        while len(self._pending) >= self._max_pending:
            self._done.append(self._pending.popleft().result())
        self._pending.append(submit(self._executor, fn, *args))

    def pop_ready(self) -> List[Any]:
        """Results of the leading jobs that have already finished (in order)."""
        out = self._done
        self._done = []
        while self._pending and (
            isinstance(self._pending[0], _InlineFuture) or self._pending[0].done()
        ):
            out.append(self._pending.popleft().result())
        return out

    def drain(self) -> List[Any]:
        out = self.pop_ready()
        while self._pending:
            out.append(self._pending.popleft().result())
        return out
