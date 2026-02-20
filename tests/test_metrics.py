"""Tests for Prometheus metrics integration."""
import pytest
from httpx import ASGITransport, AsyncClient
from prometheus_client import REGISTRY

from gateway.main import app
from gateway.metrics.collector import (
    BATCH_SIZE,
    HTTP_REQUEST_DURATION,
    HTTP_REQUESTS,
    INFERENCE_LATENCY,
    INFERENCE_REQUESTS,
    MODELS_LOADED,
)


@pytest.mark.asyncio
async def test_metrics_endpoint_returns_prometheus_format():
    """/metrics should return text in Prometheus exposition format."""
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        resp = await client.get("/metrics")

    assert resp.status_code == 200
    assert "text/plain" in resp.headers["content-type"]
    # prometheus_client always includes these built-in metrics
    assert "python_info" in resp.text


@pytest.mark.asyncio
async def test_metrics_include_custom_metrics():
    """/metrics output should contain our custom metric names."""
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        resp = await client.get("/metrics")

    body = resp.text
    assert "inference_latency_seconds" in body
    assert "inference_requests_total" in body
    assert "batch_size" in body
    assert "models_loaded_count" in body
    assert "http_request_duration_seconds" in body
    assert "http_requests_total" in body


@pytest.mark.asyncio
async def test_http_middleware_records_metrics():
    """HTTP middleware should record request duration and count for non-/metrics endpoints."""
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        # hit the health endpoint
        await client.get("/health")
        # check metrics
        resp = await client.get("/metrics")

    body = resp.text
    # middleware should have recorded a hit to /health
    # prometheus outputs labels alphabetically, so endpoint comes before method
    assert 'http_requests_total{endpoint="/health",method="GET"' in body


def test_collector_metrics_are_registered():
    """All custom metrics should be registered in the default registry."""
    metric_names = [m.name for m in REGISTRY.collect()]
    assert "inference_latency_seconds" in metric_names
    # Counters registered as "foo_total" are stored internally as "foo"
    assert "inference_requests" in metric_names
    assert "batch_size" in metric_names
    assert "models_loaded_count" in metric_names
