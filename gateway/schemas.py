from pydantic import BaseModel, Field
from typing import Any
import time


class InferenceRequest(BaseModel):
    model_id: str = Field(..., description="ID of the model to run inference against")
    inputs: list[list[float]] = Field(..., description="Batch of input vectors")


class InferenceResponse(BaseModel):
    model_id: str
    outputs: list[Any]
    latency_ms: float
    backend: str = "pytorch"
    timestamp: float = Field(default_factory=time.time)
