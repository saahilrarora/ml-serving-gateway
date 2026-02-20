"""Central registry of all Prometheus metrics.

Every metric the gateway exposes is defined here as a module-level singleton.
Other modules import what they need — this keeps definitions in one place
and avoids duplicate registration errors.
"""
from prometheus_client import Counter, Gauge, Histogram

# -- Inference metrics (recorded in predict.py) ----------------------------

INFERENCE_LATENCY = Histogram(
    "inference_latency_seconds",
    "End-to-end inference latency per request",
    labelnames=["model_id", "backend"],
    buckets=[0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5],
)

INFERENCE_REQUESTS = Counter(
    "inference_requests_total",
    "Total inference requests by model and outcome",
    labelnames=["model_id", "status"],
)

# -- Batching metrics (recorded in batcher.py) -----------------------------

BATCH_SIZE = Histogram(
    "batch_size",
    "Number of requests per batch flush",
    labelnames=["model_id"],
    buckets=[1, 2, 4, 8, 16, 32, 64],
)

# -- Model registry metrics (recorded in model_store.py) -------------------

MODELS_LOADED = Gauge(
    "models_loaded_count",
    "Number of models currently loaded in the registry",
)

# -- HTTP metrics (recorded by middleware) ----------------------------------

HTTP_REQUEST_DURATION = Histogram(
    "http_request_duration_seconds",
    "HTTP request duration by method and endpoint",
    labelnames=["method", "endpoint"],
    buckets=[0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 5.0],
)

HTTP_REQUESTS = Counter(
    "http_requests_total",
    "Total HTTP requests by method, endpoint, and status code",
    labelnames=["method", "endpoint", "status_code"],
)
