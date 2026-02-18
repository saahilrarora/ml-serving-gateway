# ML Serving Gateway

A production-grade ML model serving gateway that sits between clients and PyTorch models, solving the hard infrastructure problems of serving models at scale.

Upload a model. Get a production-ready inference endpoint — with automatic optimization, A/B routing, adaptive batching, and real-time monitoring. No restarts required.

---

## Architecture

```
                        ┌─────────────────────────────────────────────┐
                        │              ML Serving Gateway              │
                        │                                              │
 REST clients ──────────►  /predict                                   │
                        │      │                                       │
 gRPC clients ──────────►      ▼                                      │
                        │  TrafficRouter  ──► Model A (80%)           │
                        │      │          └─► Model B (20%)           │
                        │      ▼                                       │
                        │  AsyncBatcher  ← groups requests by model   │
                        │      │                                       │
                        │      ▼                                       │
                        │  ModelStore   ← PyTorch / ONNX / INT8       │
                        │                                              │
                        │  /models/*    ← load / unload at runtime    │
                        │  /routing/*   ← A/B policy management       │
                        │  /metrics     ← Prometheus scrape endpoint  │
                        └─────────────────────────────────────────────┘
                                          │
                              ┌───────────┴───────────┐
                              ▼                       ▼
                          Prometheus              Grafana
                        (time-series)          (dashboards)
```

---

## Features

| Feature | Description |
|---|---|
| **Model Registry** | Load/unload PyTorch models at runtime via API — no server restarts |
| **Adaptive Batching** | Groups concurrent requests into batches; flushes on size limit or timeout |
| **A/B Traffic Routing** | Split traffic by percentage between model versions; sticky sessions; auto-rollback on errors |
| **Inference Optimization** | On upload: auto ONNX export + INT8 quantization + benchmarking; serves fastest backend |
| **Observability** | Prometheus metrics (p50/p95/p99 latency, throughput, error rate) + Grafana dashboard |
| **REST + gRPC** | Dual protocol support; both share the same model registry and batcher |
| **Docker + Helm** | `docker-compose up` for local; Helm chart for K8s production deploy |

---

## Quickstart

**Prerequisites:** Python 3.10+, Docker, Docker Compose

```powershell
# Clone and enter
git clone https://github.com/saahilrarora/ml-serving-gateway.git
cd ml-serving-gateway

# Create and activate virtual environment
python -m venv venv
.\venv\Scripts\Activate.ps1

# Install dependencies
pip install -r requirements.txt

# Start the gateway
uvicorn gateway.main:app --reload
```

Or run the full stack (gateway + Prometheus + Grafana):

```powershell
docker-compose up
```

- Gateway: http://localhost:8000
- API docs: http://localhost:8000/docs
- Prometheus: http://localhost:9090
- Grafana: http://localhost:3000

---

## API Reference

### Health
```
GET  /health          → {"status": "ok"}
GET  /ready           → {"status": "ready", "models_loaded": 2}
```

### Inference
```
POST /predict
Body: {"model_id": "my_model", "inputs": [[0.1, 0.2, 0.3]]}
Returns: {"model_id": "my_model", "outputs": [[0.9, 0.1]], "latency_ms": 4.2, "backend": "onnx_quantized"}
```

### Model Registry
```
POST   /models/{model_id}/load          Upload a .pt file (multipart form)
GET    /models                          List all loaded models and metadata
DELETE /models/{model_id}               Unload a model from memory
GET    /models/{model_id}/optimization_status   Check optimization pipeline progress
```

### A/B Routing
```
POST   /routing/policy    Set traffic split between two model versions
GET    /routing/policy    View active routing policies
DELETE /routing/policy    Remove a routing policy
```

### Metrics
```
GET /metrics    Prometheus text format
```

---

## Build Phases

This project is built incrementally. Each phase is independently testable.

| Phase | Feature 
|---|---|
| 1 | [Core Gateway Skeleton](#phase-1-core-gateway-skeleton)
| 2 | [Model Registry](#phase-2-model-registry)
| 3 | [Adaptive Request Batching](#phase-3-adaptive-request-batching)
| 4 | [A/B Traffic Routing](#phase-4-ab-traffic-routing)
| 5 | [Inference Optimization Pipeline](#phase-5-inference-optimization-pipeline)
| 6 | [Prometheus + Grafana](#phase-6-prometheus--grafana)
| 7 | [gRPC Endpoint](#phase-7-grpc-endpoint)
| 8 | [Docker + Helm + Polish](#phase-8-docker--helm--polish)

---

### Phase 1: Core Gateway Skeleton
**Status:** ✅ Complete

A running FastAPI server with a `/predict` stub and health checks. The foundation everything else builds on.

**Files:**
- `gateway/main.py` — FastAPI app entry point with lifespan handler
- `gateway/schemas.py` — Pydantic request/response models
- `gateway/routes/health.py` — `/health` and `/ready` endpoints
- `gateway/routes/predict.py` — `/predict` (echoes input for now)
- `requirements.txt`, `Dockerfile`, `.gitignore`

**Verified:**
- `GET /health` → `{"status": "ok"}`
- `GET /ready` → `{"status": "ready", "models_loaded": 0}`
- `POST /predict` → echoes inputs back with latency tracking and metadata
- Auto-generated API docs at `/docs`

---

### Phase 2: Model Registry
**Status:** ✅ Complete

Upload a PyTorch `.pt` file, load it into memory, run inference, unload — all via API with no restarts.

**Files:**
- `gateway/registry/model_store.py` — thread-safe in-memory model store
- `gateway/registry/loader.py` — `torch.load` wrapper
- `gateway/routes/models.py` — load/list/unload endpoints
- `demo/models/text_classifier.py` — demo 2-layer text classifier
- `demo/save_demo_model.py` — serializes demo model to `.pt`

**Verified:**
- Upload `.pt` file → model loaded with detected `input_size: 10`
- `/ready` transitions from `not_ready` → `ready` when models are loaded
- `/predict` runs real PyTorch inference — 10 floats in, 3 probabilities out (0.6ms)
- `DELETE /models/{id}` unloads model and frees memory
- Duplicate load returns 409 Conflict with clear error message

---

### Phase 3: Adaptive Request Batching
**Status:** ⏳ Pending

Groups concurrent inference requests into batches. Flushes when the batch is full OR a max-wait timeout fires. Each request gets a Future resolved when its batch completes.

**Files:**
- `gateway/batching/config.py` — `BatchConfig(max_batch_size, max_wait_ms)`
- `gateway/batching/batcher.py` — `AsyncBatcher` with asyncio queue + background worker

**Test:**
```powershell
pytest tests/test_batcher.py
python demo/load_test.py --concurrency 50 --requests 500
# Logs show batch flushes; latency lower vs sequential baseline
```

---

### Phase 4: A/B Traffic Routing
**Status:** ⏳ Pending

Routes a configurable percentage of traffic to model version A vs B. Sticky sessions via IP hash. Auto-rollback when error rate exceeds threshold.

**Files:**
- `gateway/routing/policy.py` — `RoutingPolicy` dataclass
- `gateway/routing/router.py` — `TrafficRouter` with rollback logic
- `gateway/routes/routing.py` — policy CRUD endpoints

**Test:**
```powershell
curl -X POST http://localhost:8000/routing/policy -d '{"model_a":"clf_v1","model_b":"clf_v2","split":0.8,"sticky":true}'
python demo/ab_test_demo.py --requests 100
pytest tests/test_router.py
```

---

### Phase 5: Inference Optimization Pipeline
**Status:** ⏳ Pending

On model upload: export to ONNX → apply INT8 quantization → benchmark all three backends → serve the fastest. Runs in background; poll for status.

**Files:**
- `gateway/optimization/onnx_exporter.py`
- `gateway/optimization/quantizer.py`
- `gateway/optimization/benchmarker.py`
- `gateway/optimization/pipeline.py`

**Test:**
```powershell
curl http://localhost:8000/models/clf/optimization_status
# {"status":"complete","speedup":2.4,"backend":"onnx_quantized"}
pytest tests/test_optimization.py
```

---

### Phase 6: Prometheus + Grafana
**Status:** ⏳ Pending

Exposes Prometheus metrics at `/metrics`. Pre-built Grafana dashboard shows p50/p95/p99 latency, throughput, error rate, and batch size distribution in real time.

**Metrics:**
- `inference_latency_seconds` — histogram by model_id and backend
- `inference_requests_total` — counter by model_id and status
- `batch_size_total` — histogram
- `models_loaded_count` — gauge

**Test:**
```powershell
docker-compose up
# http://localhost:9090 — Prometheus targets UP
# http://localhost:3000 — Grafana dashboard live
python demo/load_test.py --concurrency 50 --requests 1000
```

---

### Phase 7: gRPC Endpoint
**Status:** ⏳ Pending

gRPC server on port 50051 runs alongside FastAPI. Shares the same model registry and batcher — no duplication.

**Files:**
- `proto/gateway.proto`
- `gateway/grpc_server.py`
- `demo/grpc_client.py`

**Test:**
```powershell
python demo/grpc_client.py --model clf_v1 --input "0.1 0.2 0.3"
pytest tests/test_grpc.py
```

---

### Phase 8: Docker + Helm + Polish
**Status:** ⏳ Pending

Full docker-compose stack. Helm chart for K8s. Second demo model (image classifier). Final README polish with benchmark results.

**Test:**
```powershell
docker-compose up --build
helm install ml-gateway ./helm
kubectl get pods
```

---

## Benchmark Results

> To be populated after Phase 3 and Phase 5 are complete.

| Scenario | Throughput | p50 Latency | p95 Latency |
|---|---|---|---|
| Sequential serving | — | — | — |
| Adaptive batching (batch=32) | — | — | — |
| ONNX optimized | — | — | — |
| ONNX + INT8 quantized | — | — | — |

---

## Project Structure

```
ml-serving-gateway/
├── gateway/
│   ├── main.py
│   ├── schemas.py
│   ├── routes/           predict.py, models.py, routing.py, health.py
│   ├── registry/         model_store.py, loader.py
│   ├── batching/         batcher.py, config.py
│   ├── routing/          router.py, policy.py
│   ├── optimization/     pipeline.py, onnx_exporter.py, quantizer.py, benchmarker.py
│   ├── metrics/          collector.py, middleware.py
│   └── grpc_server.py
├── proto/
│   └── gateway.proto
├── demo/
│   ├── models/           text_classifier.py, image_classifier.py
│   ├── save_demo_model.py
│   ├── load_test.py
│   ├── ab_test_demo.py
│   └── grpc_client.py
├── tests/
│   ├── test_predict.py
│   ├── test_registry.py
│   ├── test_batcher.py
│   ├── test_router.py
│   ├── test_optimization.py
│   └── test_grpc.py
├── grafana/
│   └── dashboard.json
├── helm/
│   ├── Chart.yaml
│   ├── values.yaml
│   └── templates/
├── docker-compose.yml
├── Dockerfile
├── prometheus.yml
├── requirements.txt
└── README.md
```

---

## Dependencies

```
fastapi uvicorn[standard] pydantic   # REST server
torch torchvision                    # model loading + inference
onnx onnxruntime                     # export and optimized inference
grpcio grpcio-tools                  # gRPC
prometheus_client                    # metrics
httpx pytest pytest-asyncio          # testing
```
