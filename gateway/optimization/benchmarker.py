"""Benchmark inference latency across backends."""
import logging
import time

import torch

logger = logging.getLogger(__name__)


def benchmark(callable_model, sample_input: torch.Tensor,
              warmup: int = 10, iterations: int = 100) -> dict:
    """Run timed inferences and return p50/p95/p99 latency in ms.

    Works with any callable (PyTorch module, OnnxModelWrapper) — the caller
    doesn't need to know what backend is being benchmarked.
    """
    # warmup: let JIT/caches settle before measuring
    with torch.no_grad():
        for _ in range(warmup):
            callable_model(sample_input)

    # timed runs
    latencies = []
    with torch.no_grad():
        for _ in range(iterations):
            start = time.perf_counter()
            callable_model(sample_input)
            elapsed_ms = (time.perf_counter() - start) * 1000
            latencies.append(elapsed_ms)

    latencies.sort()
    n = len(latencies)

    stats = {
        "p50_ms": round(latencies[n // 2], 3),
        "p95_ms": round(latencies[int(n * 0.95)], 3),
        "p99_ms": round(latencies[int(n * 0.99)], 3),
    }

    logger.info("Benchmark (%d iters): p50=%.3fms p95=%.3fms p99=%.3fms",
                iterations, stats["p50_ms"], stats["p95_ms"], stats["p99_ms"])
    return stats
