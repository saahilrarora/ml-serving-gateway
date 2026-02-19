import logging

from fastapi import APIRouter, HTTPException, Request

from gateway.routing.policy import RoutingPolicy
from gateway.schemas import RoutingPolicyRequest

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/routing", tags=["routing"])


@router.post("/policy")
async def create_policy(body: RoutingPolicyRequest, request: Request):
    """Create or update a routing policy for A/B traffic splitting."""
    traffic_router = request.app.state.router

    policy = RoutingPolicy(
        name=body.name,
        model_a=body.model_a,
        model_b=body.model_b,
        traffic_split=body.traffic_split,
        sticky=body.sticky,
        error_threshold=body.error_threshold,
        min_requests=body.min_requests,
    )
    traffic_router.add_policy(policy)

    return {
        "status": "created",
        "policy": {
            "name": policy.name,
            "model_a": policy.model_a,
            "model_b": policy.model_b,
            "traffic_split": policy.traffic_split,
            "sticky": policy.sticky,
            "error_threshold": policy.error_threshold,
            "min_requests": policy.min_requests,
        },
    }


@router.get("/policy")
async def list_policies(request: Request):
    """List all active routing policies."""
    traffic_router = request.app.state.router
    policies = traffic_router.list_policies()
    return {"policies": policies, "count": len(policies)}


@router.delete("/policy/{name}")
async def delete_policy(name: str, request: Request):
    """Remove a routing policy. Traffic for that name reverts to direct model lookup."""
    traffic_router = request.app.state.router
    removed = traffic_router.remove_policy(name)

    if not removed:
        raise HTTPException(status_code=404, detail=f"No routing policy named '{name}'")

    return {"name": name, "status": "deleted"}
