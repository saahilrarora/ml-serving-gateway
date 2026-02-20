import asyncio
import logging
import time
from dataclasses import dataclass, field

import torch

from gateway.batching.config import BatchConfig
from gateway.metrics.collector import BATCH_SIZE
from gateway.registry.model_store import ModelStore

logger = logging.getLogger(__name__)


@dataclass
class _BatchItem:
    """One queued inference request: its tensor + a Future for the caller."""

    model_id: str
    input_tensor: torch.Tensor
    future: asyncio.Future
    enqueued_at: float = field(default_factory=time.monotonic)


class AsyncBatcher:
    """Accumulates per-model inference requests and flushes them as batches."""

    def __init__(self, model_store: ModelStore, config: BatchConfig | None = None):
        self._store = model_store
        self._config = config or BatchConfig()
        self._queues: dict[str, list[_BatchItem]] = {}   # model_id → pending items
        self._lock = asyncio.Lock()
        self._event = asyncio.Event()                     # wakes worker on new item
        self._worker_task: asyncio.Task | None = None

    # -- Lifecycle ---------------------------------------------------------

    async def start(self) -> None:
        self._worker_task = asyncio.create_task(self._worker())
        logger.info(
            "AsyncBatcher started (max_batch=%d, max_wait=%.1fms)",
            self._config.max_batch_size,
            self._config.max_wait_ms,
        )

    async def stop(self) -> None:
        if self._worker_task:
            self._worker_task.cancel()
            try:
                await self._worker_task
            except asyncio.CancelledError:
                pass

        # Reject anything still queued so callers don't hang
        async with self._lock:
            for items in self._queues.values():
                for item in items:
                    if not item.future.done():
                        item.future.set_exception(RuntimeError("Batcher shutting down"))
                items.clear()

        logger.info("AsyncBatcher stopped")

    # -- Public API --------------------------------------------------------

    async def submit(self, model_id: str, input_tensor: torch.Tensor) -> list:
        """Add a request to the queue and block until its batch is processed."""
        loop = asyncio.get_running_loop()
        future: asyncio.Future = loop.create_future()

        async with self._lock:
            if model_id not in self._queues:
                self._queues[model_id] = []
            self._queues[model_id].append(
                _BatchItem(model_id=model_id, input_tensor=input_tensor, future=future)
            )

        self._event.set()       # wake the worker
        return await future     # block until batch completes

    # -- Background worker -------------------------------------------------

    async def _worker(self) -> None:
        """Loop: wait for signal/timeout → check queues → flush ready batches."""
        timeout_sec = self._config.max_wait_ms / 1000.0

        while True:
            # Sleep until a new item arrives or the timeout expires
            try:
                await asyncio.wait_for(self._event.wait(), timeout=timeout_sec)
            except asyncio.TimeoutError:
                pass
            self._event.clear()

            # Decide which queues to flush (under lock), then flush outside lock
            to_flush: list[tuple[str, list[_BatchItem]]] = []

            async with self._lock:
                now = time.monotonic()
                for model_id, items in list(self._queues.items()):
                    if not items:
                        continue

                    oldest_wait_ms = (now - items[0].enqueued_at) * 1000

                    if len(items) >= self._config.max_batch_size:
                        # Size trigger: take max_batch_size, leave the rest
                        batch = items[: self._config.max_batch_size]
                        self._queues[model_id] = items[self._config.max_batch_size :]
                        to_flush.append((model_id, batch))
                        logger.info(
                            "[batcher] flushing batch of %d for '%s' (size trigger)",
                            len(batch), model_id,
                        )
                    elif oldest_wait_ms >= self._config.max_wait_ms:
                        # Timeout trigger: flush everything in queue
                        to_flush.append((model_id, list(items)))
                        items.clear()
                        logger.info(
                            "[batcher] timeout flush of %d for '%s' (%.1fms waited)",
                            len(to_flush[-1][1]), model_id, oldest_wait_ms,
                        )

            for model_id, batch in to_flush:
                await self._flush(model_id, batch)

    async def _flush(self, model_id: str, batch: list[_BatchItem]) -> None:
        """Stack tensors → one forward pass → split results back to callers."""
        BATCH_SIZE.labels(model_id=model_id).observe(len(batch))
        try:
            model_info = await self._store.get(model_id)
            if not model_info:
                exc = RuntimeError(f"Model '{model_id}' not found (unloaded?)")
                for item in batch:
                    if not item.future.done():
                        item.future.set_exception(exc)
                return

            # e.g. 8 tensors of shape (10,) → one tensor of shape (8, 10)
            stacked = torch.stack([item.input_tensor for item in batch])

            with torch.no_grad():
                outputs = model_info.model(stacked)

            # Fan results back — outputs[i] goes to batch[i]'s caller
            for item, output in zip(batch, outputs):
                if not item.future.done():
                    item.future.set_result(output.tolist())

        except Exception as exc:
            for item in batch:
                if not item.future.done():
                    item.future.set_exception(exc)
