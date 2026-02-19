"""Tests for the AsyncBatcher.

Uses the real TextClassifier model (not mocks) to verify end-to-end
batching behavior: size triggers, timeout triggers, multi-model queues,
missing-model errors, and concurrent correctness.
"""

import asyncio
import time

import pytest
import pytest_asyncio
import torch

from demo.models.text_classifier import TextClassifier
from gateway.batching.batcher import AsyncBatcher
from gateway.batching.config import BatchConfig
from gateway.registry.model_store import ModelStore


@pytest_asyncio.fixture
async def store():
    """Model store with one TextClassifier loaded."""
    s = ModelStore()
    model = TextClassifier(input_size=10, num_classes=3)
    model.eval()
    await s.add("clf", model, input_size=10)
    return s


@pytest_asyncio.fixture
async def batcher(store):
    """AsyncBatcher with a small batch size and short timeout for testing."""
    b = AsyncBatcher(store, BatchConfig(max_batch_size=4, max_wait_ms=30))
    await b.start()
    yield b
    await b.stop()


def _random_input():
    """Single sample: shape (10,)."""
    return torch.randn(10)


# -- Tests -----------------------------------------------------------------


@pytest.mark.asyncio
async def test_single_request(batcher):
    """One request should flush after the timeout and return 3 probabilities."""
    result = await batcher.submit("clf", _random_input())
    assert isinstance(result, list)
    assert len(result) == 3             # TextClassifier outputs 3 classes
    assert abs(sum(result) - 1.0) < 0.01  # softmax sums to ~1


@pytest.mark.asyncio
async def test_flush_on_full(batcher):
    """Submitting max_batch_size items should flush immediately (no timeout wait)."""
    start = time.monotonic()

    # Fire 4 requests concurrently (max_batch_size=4)
    tasks = [batcher.submit("clf", _random_input()) for _ in range(4)]
    results = await asyncio.gather(*tasks)

    elapsed_ms = (time.monotonic() - start) * 1000

    assert len(results) == 4
    for r in results:
        assert len(r) == 3

    # Should flush well before the 30ms timeout since batch is full
    assert elapsed_ms < 30, f"Took {elapsed_ms:.1f}ms — expected size-triggered flush"


@pytest.mark.asyncio
async def test_flush_on_timeout(batcher):
    """Fewer than max_batch_size items should flush after the timeout."""
    start = time.monotonic()

    # Only 2 items (max_batch_size=4), so must wait for timeout
    tasks = [batcher.submit("clf", _random_input()) for _ in range(2)]
    results = await asyncio.gather(*tasks)

    elapsed_ms = (time.monotonic() - start) * 1000

    assert len(results) == 2
    # Should have waited roughly max_wait_ms (30ms) before flushing
    assert elapsed_ms >= 25, f"Flushed too early at {elapsed_ms:.1f}ms"


@pytest.mark.asyncio
async def test_multiple_models(store):
    """Requests for different models are batched independently."""
    # Load a second model
    model2 = TextClassifier(input_size=10, num_classes=3)
    model2.eval()
    await store.add("clf_v2", model2, input_size=10)

    batcher = AsyncBatcher(store, BatchConfig(max_batch_size=4, max_wait_ms=30))
    await batcher.start()

    try:
        tasks = [
            batcher.submit("clf", _random_input()),
            batcher.submit("clf_v2", _random_input()),
            batcher.submit("clf", _random_input()),
            batcher.submit("clf_v2", _random_input()),
        ]
        results = await asyncio.gather(*tasks)
        assert len(results) == 4
        for r in results:
            assert len(r) == 3
    finally:
        await batcher.stop()


@pytest.mark.asyncio
async def test_model_not_found(batcher):
    """Submitting for a non-existent model should raise RuntimeError."""
    with pytest.raises(RuntimeError, match="not found"):
        await batcher.submit("nonexistent", _random_input())


@pytest.mark.asyncio
async def test_concurrent_correctness(batcher):
    """Many concurrent requests should all complete without errors."""
    tasks = [batcher.submit("clf", _random_input()) for _ in range(20)]
    results = await asyncio.gather(*tasks)

    assert len(results) == 20
    for r in results:
        assert len(r) == 3
        assert abs(sum(r) - 1.0) < 0.01
