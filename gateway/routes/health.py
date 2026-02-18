from fastapi import APIRouter, Request

router = APIRouter()


@router.get("/health")
async def health():
    return {"status": "ok"}


@router.get("/ready")
async def ready(request: Request):
    # query the real model store to report how many models are loaded
    store = request.app.state.model_store
    count = await store.count()
    return {"status": "ready" if count > 0 else "not_ready", "models_loaded": count}
