import time

import torch
from fastapi import APIRouter, HTTPException, Request

from gateway.schemas import InferenceRequest, InferenceResponse

router = APIRouter()


@router.post("/predict", response_model=InferenceResponse)
async def predict(inference_req: InferenceRequest, request: Request):
    start = time.perf_counter()

    # look up the model in the registry
    store = request.app.state.model_store
    model_info = await store.get(inference_req.model_id)
    if not model_info:
        raise HTTPException(status_code=404, detail=f"Model '{inference_req.model_id}' not found. Load it first via POST /models/{{model_id}}/load")

    # convert input list → PyTorch tensor for the forward pass
    input_tensor = torch.tensor(inference_req.inputs, dtype=torch.float32)

    # run inference (no_grad disables gradient tracking — saves memory, faster)
    with torch.no_grad():
        output_tensor = model_info.model(input_tensor)

    # convert output tensor back to a Python list for the JSON response
    outputs = output_tensor.tolist()

    latency_ms = (time.perf_counter() - start) * 1000

    return InferenceResponse(
        model_id=inference_req.model_id,
        outputs=outputs,
        latency_ms=round(latency_ms, 3),
        backend="pytorch",
    )
