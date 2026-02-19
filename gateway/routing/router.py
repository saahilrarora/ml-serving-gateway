import hashlib
import logging
import random

from gateway.routing.policy import RoutingPolicy

logger = logging.getLogger(__name__)


class TrafficRouter:
    """Resolves which model version handles a request based on routing policies."""

    def __init__(self):
        self._policies: dict[str, RoutingPolicy] = {}   # name → policy
        self._error_counts: dict[str, int] = {}          # model_id → error count
        self._request_counts: dict[str, int] = {}        # model_id → total requests
        self._rolled_back: set[str] = set()              # policy names that triggered rollback

    # -- Policy CRUD -------------------------------------------------------

    def add_policy(self, policy: RoutingPolicy) -> None:
        self._policies[policy.name] = policy
        # initialize counters for both models
        for model_id in (policy.model_a, policy.model_b):
            self._error_counts.setdefault(model_id, 0)
            self._request_counts.setdefault(model_id, 0)
        logger.info(
            "Routing policy '%s': %s (%.0f%%) / %s (%.0f%%)",
            policy.name, policy.model_a, policy.traffic_split * 100,
            policy.model_b, (1 - policy.traffic_split) * 100,
        )

    def get_policy(self, name: str) -> RoutingPolicy | None:
        return self._policies.get(name)

    def remove_policy(self, name: str) -> bool:
        if name in self._policies:
            del self._policies[name]
            self._rolled_back.discard(name)
            return True
        return False

    def list_policies(self) -> list[dict]:
        results = []
        for p in self._policies.values():
            results.append({
                "name": p.name,
                "model_a": p.model_a,
                "model_b": p.model_b,
                "traffic_split": p.traffic_split,
                "sticky": p.sticky,
                "error_threshold": p.error_threshold,
                "min_requests": p.min_requests,
                "rolled_back": p.name in self._rolled_back,
            })
        return results

    # -- Routing decision --------------------------------------------------

    def resolve(self, model_id: str, client_ip: str | None = None) -> str:
        """Pick which model version should handle this request.

        If no policy exists for this model_id, returns it unchanged.
        This keeps the router backward-compatible with Phase 2/3 behavior.
        """
        policy = self._policies.get(model_id)
        if not policy:
            return model_id

        # if rollback triggered, all traffic goes to model_a
        if policy.name in self._rolled_back:
            return policy.model_a

        if policy.sticky and client_ip:
            # hash the IP to get a stable float in [0, 1)
            ip_hash = int(hashlib.md5(client_ip.encode()).hexdigest(), 16)
            score = (ip_hash % 10000) / 10000.0
        else:
            score = random.random()

        chosen = policy.model_a if score < policy.traffic_split else policy.model_b
        self._request_counts[chosen] = self._request_counts.get(chosen, 0) + 1
        return chosen

    # -- Error tracking + auto-rollback ------------------------------------

    def record_success(self, model_id: str) -> None:
        self._request_counts[model_id] = self._request_counts.get(model_id, 0) + 1

    def record_error(self, model_id: str) -> None:
        self._error_counts[model_id] = self._error_counts.get(model_id, 0) + 1
        self._request_counts[model_id] = self._request_counts.get(model_id, 0) + 1
        self._check_rollback(model_id)

    def _check_rollback(self, model_id: str) -> None:
        """If model_b's error rate exceeds the threshold, roll back to model_a."""
        for policy in self._policies.values():
            if policy.model_b != model_id:
                continue
            if policy.name in self._rolled_back:
                continue

            total = self._request_counts.get(model_id, 0)
            errors = self._error_counts.get(model_id, 0)
            if total < policy.min_requests:
                continue

            error_rate = errors / total
            if error_rate > policy.error_threshold:
                self._rolled_back.add(policy.name)
                logger.warning(
                    "AUTO-ROLLBACK: policy '%s' — model_b '%s' error rate %.1f%% "
                    "exceeds threshold %.1f%%. All traffic → '%s'",
                    policy.name, model_id, error_rate * 100,
                    policy.error_threshold * 100, policy.model_a,
                )
