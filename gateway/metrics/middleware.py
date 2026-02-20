"""FastAPI middleware for HTTP-level Prometheus metrics."""
import time

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request

from gateway.metrics.collector import HTTP_REQUEST_DURATION, HTTP_REQUESTS


class PrometheusMiddleware(BaseHTTPMiddleware):
    """Records request duration and count for every HTTP request.

    These are general HTTP metrics (by method/endpoint/status_code),
    separate from the inference-specific metrics in predict.py.
    Skips the /metrics endpoint itself to avoid self-referential noise.
    """

    async def dispatch(self, request: Request, call_next):
        # don't instrument the metrics endpoint itself
        if request.url.path == "/metrics":
            return await call_next(request)

        start = time.perf_counter()
        response = await call_next(request)
        duration = time.perf_counter() - start

        endpoint = request.url.path
        method = request.method
        status = str(response.status_code)

        HTTP_REQUEST_DURATION.labels(method=method, endpoint=endpoint).observe(duration)
        HTTP_REQUESTS.labels(method=method, endpoint=endpoint, status_code=status).inc()

        return response
