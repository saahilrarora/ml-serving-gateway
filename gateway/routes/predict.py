import time

from fastapi import APIRouter

from gateway.schemas import InferenceRequest, InferenceResponse

router = APIRouter()


@router.post("/predict", response_model=InferenceResponse)
async def predict(request: InferenceRequest):
    start = time.perf_counter()

    # Phase 1: echo inputs back as outputs (stub)
    outputs = request.inputs

    latency_ms = (time.perf_counter() - start) * 1000

    return InferenceResponse(
        model_id=request.model_id,
        outputs=outputs,
        latency_ms=round(latency_ms, 3),
        backend="echo",
    )
