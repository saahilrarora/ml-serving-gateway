"""
A/B test demo for the ML Serving Gateway.

Sends many requests through a routing policy and prints how many
went to each model version. Expects the server to be running with
two model versions loaded and a routing policy active.

Setup before running:
    curl.exe -X POST http://localhost:8000/models/clf_v1/load -F "file=@demo/text_classifier.pt"
    curl.exe -X POST http://localhost:8000/models/clf_v2/load -F "file=@demo/text_classifier.pt"
    curl.exe -X POST http://localhost:8000/routing/policy -H "Content-Type: application/json" ^
        -d '{"name":"clf","model_a":"clf_v1","model_b":"clf_v2","traffic_split":0.8}'

Usage:
    python demo/ab_test_demo.py --requests 100
"""

import argparse
import asyncio
from collections import Counter

import httpx

PAYLOAD = {
    "model_id": "clf",
    "inputs": [[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]],
}


async def run_ab_test(base_url: str, total: int):
    url = f"{base_url}/predict"
    model_counts: Counter = Counter()
    errors = 0

    async with httpx.AsyncClient(timeout=30.0) as client:
        for i in range(total):
            try:
                resp = await client.post(url, json=PAYLOAD)
                resp.raise_for_status()
                data = resp.json()
                model_counts[data["model_id"]] += 1
            except Exception as e:
                errors += 1
                print(f"  ERROR on request {i}: {e}")

    print("\n" + "=" * 45)
    print("  A/B Test Results")
    print("=" * 45)
    print(f"  Total requests : {total}")
    print(f"  Errors         : {errors}")
    for model_id, count in model_counts.most_common():
        pct = count / total * 100
        print(f"  {model_id:15s} : {count:4d}  ({pct:.1f}%)")
    print("=" * 45 + "\n")


def main():
    parser = argparse.ArgumentParser(description="A/B routing demo")
    parser.add_argument("--base-url", default="http://localhost:8000")
    parser.add_argument("--requests", type=int, default=100)
    args = parser.parse_args()

    print(f"Sending {args.requests} requests to {args.base_url}/predict")
    print(f"model_id in payload: '{PAYLOAD['model_id']}'")
    asyncio.run(run_ab_test(args.base_url, args.requests))


if __name__ == "__main__":
    main()
