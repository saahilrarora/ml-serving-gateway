import time

import torch
from fastapi import APIRouter, HTTPException, Request

from gateway.schemas import InferenceRequest, InferenceResponse

router = APIRouter()


@router.post("/predict", response_model=InferenceResponse)
async def predict(inference_req: InferenceRequest, request: Request):
    start = time.perf_counter()

    # resolve model_id through the traffic router — if an A/B policy exists
    # for this model_id, the router picks the actual model version
    traffic_router = request.app.state.router
    client_ip = request.client.host if request.client else None
    resolved_model_id = traffic_router.resolve(inference_req.model_id, client_ip)

    # convert input list → tensor (each row is one sample)
    input_tensor = torch.tensor(inference_req.inputs, dtype=torch.float32)

    # submit to the batcher with the resolved model version
    batcher = request.app.state.batcher
    try:
        outputs = await batcher.submit(resolved_model_id, input_tensor)
        traffic_router.record_success(resolved_model_id)
    except RuntimeError as e:
        traffic_router.record_error(resolved_model_id)
        raise HTTPException(status_code=404, detail=str(e))

    latency_ms = (time.perf_counter() - start) * 1000

    # look up which backend is actually serving this model
    store = request.app.state.model_store
    model_info = await store.get(resolved_model_id)
    backend = model_info.backend if model_info else "pytorch"

    return InferenceResponse(
        model_id=resolved_model_id,
        outputs=outputs,
        latency_ms=round(latency_ms, 3),
        backend=backend,
    )
