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


# -- Routing schemas -------------------------------------------------------

class RoutingPolicyRequest(BaseModel):
    """Request body for creating/updating a routing policy."""
    name: str = Field(..., description="Policy name — the model_id clients use (e.g. 'clf')")
    model_a: str = Field(..., description="Primary model version ID")
    model_b: str = Field(..., description="Canary / new model version ID")
    traffic_split: float = Field(0.8, ge=0.0, le=1.0, description="Fraction of traffic → model_a")
    sticky: bool = Field(True, description="Same client IP always hits the same version")
    error_threshold: float = Field(0.5, ge=0.0, le=1.0, description="Error rate that triggers rollback")
    min_requests: int = Field(5, ge=1, description="Minimum requests before error rate is evaluated")
