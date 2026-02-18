import logging

from fastapi import APIRouter, File, HTTPException, Request, UploadFile

from gateway.registry.loader import load_model_from_bytes

logger = logging.getLogger(__name__)

router = APIRouter()


@router.post("/models/{model_id}/load")
async def load_model(model_id: str, request: Request, file: UploadFile = File(...)):
    """Upload and load a .pt model file into the registry."""
    store = request.app.state.model_store  # shared ModelStore instance from main.py

    # check if a model with this ID is already loaded
    existing = await store.get(model_id)
    if existing:
        raise HTTPException(status_code=409, detail=f"Model '{model_id}' is already loaded. Unload it first.")

    # read the uploaded file bytes
    model_bytes = await file.read()
    logger.info(f"Received model '{model_id}' ({len(model_bytes)} bytes)")

    # deserialize the PyTorch model from the bytes
    try:
        model, input_size = load_model_from_bytes(model_bytes)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to load model: {e}")

    # register it in the store
    info = await store.add(model_id, model, input_size)

    return {
        "model_id": info.model_id,
        "input_size": info.input_size,
        "loaded_at": info.loaded_at,
        "status": "loaded",
    }


@router.get("/models")
async def list_models(request: Request):
    """List all currently loaded models."""
    store = request.app.state.model_store
    models = await store.list_models()
    return {"models": models, "count": len(models)}


@router.delete("/models/{model_id}")
async def unload_model(model_id: str, request: Request):
    """Unload a model from memory."""
    store = request.app.state.model_store
    removed = await store.remove(model_id)

    if not removed:
        raise HTTPException(status_code=404, detail=f"Model '{model_id}' not found")

    logger.info(f"Model '{model_id}' unloaded")
    return {"model_id": model_id, "status": "unloaded"}
