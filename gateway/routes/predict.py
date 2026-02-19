import time

import torch
from fastapi import APIRouter, HTTPException, Request

from gateway.schemas import InferenceRequest, InferenceResponse

router = APIRouter()


@router.post("/predict", response_model=InferenceResponse)
async def predict(inference_req: InferenceRequest, request: Request):
    start = time.perf_counter()

    # convert input list → tensor (each row is one sample)
    input_tensor = torch.tensor(inference_req.inputs, dtype=torch.float32)

    # submit to the batcher — blocks until our batch is flushed
    batcher = request.app.state.batcher
    try:
        outputs = await batcher.submit(inference_req.model_id, input_tensor)
    except RuntimeError as e:
        raise HTTPException(status_code=404, detail=str(e))

    latency_ms = (time.perf_counter() - start) * 1000

    return InferenceResponse(
        model_id=inference_req.model_id,
        outputs=outputs,
        latency_ms=round(latency_ms, 3),
        backend="pytorch",
    )
