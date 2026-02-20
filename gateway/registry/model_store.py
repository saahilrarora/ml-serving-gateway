import asyncio
import os
import shutil
import time
from dataclasses import dataclass, field
from typing import Any

from gateway.metrics.collector import MODELS_LOADED


@dataclass
class ModelInfo:
    """Metadata and reference for a loaded model."""
    model: Any                    # callable: PyTorch module or OnnxModelWrapper
    model_id: str                 # unique identifier, e.g. "text_clf"
    input_size: int               # expected input dimension
    loaded_at: float = field(default_factory=time.time)
    backend: str = "pytorch"      # "pytorch", "onnx", or "onnx_quantized"
    optimization_status: dict = field(default_factory=lambda: {"status": "none"})
    artifact_dir: str | None = None  # temp dir holding ONNX files, cleaned on unload


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
            MODELS_LOADED.inc()
            return info

    async def get(self, model_id: str) -> ModelInfo | None:
        """Look up a model by ID. Returns None if not found."""
        async with self._lock:
            return self._models.get(model_id)

    async def remove(self, model_id: str) -> bool:
        """Unload a model and clean up any optimization artifacts."""
        async with self._lock:
            if model_id in self._models:
                info = self._models.pop(model_id)
                MODELS_LOADED.dec()
                # clean up ONNX files on disk
                if info.artifact_dir and os.path.exists(info.artifact_dir):
                    shutil.rmtree(info.artifact_dir, ignore_errors=True)
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
                    "backend": info.backend,
                    "optimization_status": info.optimization_status["status"],
                }
                for info in self._models.values()
            ]

    async def count(self) -> int:
        """How many models are currently loaded."""
        async with self._lock:
            return len(self._models)

    async def update_backend(self, model_id: str, model: Any, backend: str) -> bool:
        """Swap the model callable and backend after optimization picks a winner."""
        async with self._lock:
            info = self._models.get(model_id)
            if not info:
                return False
            info.model = model
            info.backend = backend
            return True

    async def set_optimization_status(self, model_id: str, status: dict) -> bool:
        """Update the optimization status dict for a model."""
        async with self._lock:
            info = self._models.get(model_id)
            if not info:
                return False
            info.optimization_status = status
            return True

    async def set_artifact_dir(self, model_id: str, artifact_dir: str) -> bool:
        """Record the temp directory where ONNX artifacts live."""
        async with self._lock:
            info = self._models.get(model_id)
            if not info:
                return False
            info.artifact_dir = artifact_dir
            return True
