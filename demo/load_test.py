"""
Load test for the ML Serving Gateway.

Fires many concurrent /predict requests using httpx and measures
throughput and latency percentiles. Expects a model to be loaded first.

Usage:
    python demo/load_test.py --concurrency 50 --requests 500
    python demo/load_test.py --sequential          # baseline without concurrency
"""

import argparse
import asyncio
import statistics
import time

import httpx

# Default payload — matches the demo TextClassifier (10 floats in)
PAYLOAD = {
    "model_id": "text_clf",
    "inputs": [[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]],
}


async def send_request(client: httpx.AsyncClient, url: str) -> float:
    """Send one POST /predict and return the round-trip latency in ms."""
    start = time.perf_counter()
    resp = await client.post(url, json=PAYLOAD)
    elapsed = (time.perf_counter() - start) * 1000
    resp.raise_for_status()
    return elapsed


async def run_concurrent(base_url: str, total: int, concurrency: int):
    """Fire requests with bounded concurrency using a semaphore."""
    url = f"{base_url}/predict"
    semaphore = asyncio.Semaphore(concurrency)
    latencies: list[float] = []
    errors = 0

    async with httpx.AsyncClient(timeout=30.0) as client:

        async def _task():
            nonlocal errors
            async with semaphore:
                try:
                    lat = await send_request(client, url)
                    latencies.append(lat)
                except Exception as e:
                    errors += 1
                    print(f"  ERROR: {e}")

        wall_start = time.perf_counter()
        await asyncio.gather(*[_task() for _ in range(total)])
        wall_elapsed = time.perf_counter() - wall_start

    _print_results(latencies, errors, wall_elapsed, total, concurrency)


async def run_sequential(base_url: str, total: int):
    """Fire requests one at a time (baseline comparison)."""
    url = f"{base_url}/predict"
    latencies: list[float] = []
    errors = 0

    async with httpx.AsyncClient(timeout=30.0) as client:
        wall_start = time.perf_counter()
        for _ in range(total):
            try:
                lat = await send_request(client, url)
                latencies.append(lat)
            except Exception as e:
                errors += 1
                print(f"  ERROR: {e}")
        wall_elapsed = time.perf_counter() - wall_start

    _print_results(latencies, errors, wall_elapsed, total, concurrency=1)


def _print_results(latencies, errors, wall_sec, total, concurrency):
    """Print throughput and latency percentiles."""
    latencies.sort()
    n = len(latencies)

    print("\n" + "=" * 55)
    print(f"  Load Test Results  (concurrency={concurrency})")
    print("=" * 55)
    print(f"  Total requests : {total}")
    print(f"  Successful     : {n}")
    print(f"  Errors         : {errors}")
    print(f"  Wall time      : {wall_sec:.2f}s")
    print(f"  Throughput     : {n / wall_sec:.1f} req/s")

    if n > 0:
        p50 = latencies[int(n * 0.50)]
        p95 = latencies[int(n * 0.95)]
        p99 = latencies[min(int(n * 0.99), n - 1)]
        print(f"  Mean latency   : {statistics.mean(latencies):.2f}ms")
        print(f"  p50 latency    : {p50:.2f}ms")
        print(f"  p95 latency    : {p95:.2f}ms")
        print(f"  p99 latency    : {p99:.2f}ms")
    print("=" * 55 + "\n")


def main():
    parser = argparse.ArgumentParser(description="Load test the ML Serving Gateway")
    parser.add_argument("--base-url", default="http://localhost:8000")
    parser.add_argument("--requests", type=int, default=500, help="Total requests to send")
    parser.add_argument("--concurrency", type=int, default=50, help="Max concurrent requests")
    parser.add_argument("--sequential", action="store_true", help="Run sequentially (baseline)")
    args = parser.parse_args()

    print(f"Target: {args.base_url}")
    print(f"Requests: {args.requests}, Concurrency: {'1 (sequential)' if args.sequential else args.concurrency}")

    if args.sequential:
        asyncio.run(run_sequential(args.base_url, args.requests))
    else:
        asyncio.run(run_concurrent(args.base_url, args.requests, args.concurrency))


if __name__ == "__main__":
    main()
