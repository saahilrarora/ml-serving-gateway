import asyncio
import time
from dataclasses import dataclass, field
from typing import Any


@dataclass
class ModelInfo:
    """Metadata and reference for a loaded model."""
    model: Any                    # the actual PyTorch model object
    model_id: str                 # unique identifier, e.g. "text_clf"
    input_size: int               # expected input dimension
    loaded_at: float = field(default_factory=time.time)  # timestamp of when it was loaded


class ModelStore:
    """Thread-safe in-memory registry of loaded models.
    Uses asyncio.Lock so multiple async requests don't corrupt the dict."""

    def __init__(self):
        self._models: dict[str, ModelInfo] = {}  # model_id → ModelInfo
        self._lock = asyncio.Lock()               # protects concurrent access

    async def add(self, model_id: str, model: Any, input_size: int) -> ModelInfo:
        """Register a loaded model in the store."""
        async with self._lock:
            info = ModelInfo(model=model, model_id=model_id, input_size=input_size)
            self._models[model_id] = info
            return info

    async def get(self, model_id: str) -> ModelInfo | None:
        """Look up a model by ID. Returns None if not found."""
        async with self._lock:
            return self._models.get(model_id)

    async def remove(self, model_id: str) -> bool:
        """Unload a model. Returns True if it existed, False otherwise."""
        async with self._lock:
            if model_id in self._models:
                del self._models[model_id]  # removes reference; Python GC frees memory
                return True
            return False

    async def list_models(self) -> list[dict]:
        """Return metadata for all loaded models (without the model objects themselves)."""
        async with self._lock:
            return [
                {
                    "model_id": info.model_id,
                    "input_size": info.input_size,
                    "loaded_at": info.loaded_at,
                }
                for info in self._models.values()
            ]

    async def count(self) -> int:
        """How many models are currently loaded."""
        async with self._lock:
            return len(self._models)
