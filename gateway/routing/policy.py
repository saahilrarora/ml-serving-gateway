from dataclasses import dataclass


@dataclass
class RoutingPolicy:
    """Describes how to split traffic between two model versions.

    Example: name="clf", model_a="clf_v1", model_b="clf_v2", traffic_split=0.8
    means requests for "clf" go 80% to clf_v1 and 20% to clf_v2.
    """

    name: str                        # the model_id clients use (e.g. "clf")
    model_a: str                     # primary model version
    model_b: str                     # canary / new model version
    traffic_split: float = 0.8       # fraction of traffic → model_a (0.0–1.0)
    sticky: bool = True              # same client IP always hits the same version
    error_threshold: float = 0.5     # if model_b error rate exceeds this, rollback
    min_requests: int = 5            # minimum requests before error rate is evaluated
