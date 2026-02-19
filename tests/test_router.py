"""Tests for the TrafficRouter.

Validates routing decisions: traffic splits, sticky sessions, auto-rollback,
and pass-through when no policy exists.
"""

import pytest

from gateway.routing.policy import RoutingPolicy
from gateway.routing.router import TrafficRouter


@pytest.fixture
def router():
    return TrafficRouter()


@pytest.fixture
def policy():
    """80/20 split, sticky sessions, rollback after 50% errors on 5+ requests."""
    return RoutingPolicy(
        name="clf",
        model_a="clf_v1",
        model_b="clf_v2",
        traffic_split=0.8,
        sticky=True,
        error_threshold=0.5,
        min_requests=5,
    )


# -- Tests -----------------------------------------------------------------


def test_no_policy_passthrough(router):
    """Without a policy, the model_id should pass through unchanged."""
    assert router.resolve("some_model") == "some_model"
    assert router.resolve("another_model", client_ip="1.2.3.4") == "another_model"


def test_traffic_split_distribution(router, policy):
    """Over many non-sticky requests, traffic should roughly match the split."""
    policy.sticky = False
    router.add_policy(policy)

    counts = {"clf_v1": 0, "clf_v2": 0}
    for _ in range(1000):
        chosen = router.resolve("clf")
        counts[chosen] += 1

    # 80/20 split — allow some variance
    ratio = counts["clf_v1"] / 1000
    assert 0.7 < ratio < 0.9, f"Expected ~80% to clf_v1, got {ratio:.1%}"


def test_sticky_sessions(router, policy):
    """Same IP should always resolve to the same model version."""
    router.add_policy(policy)

    # resolve many times for the same IP — should always get the same result
    results = {router.resolve("clf", client_ip="192.168.1.42") for _ in range(50)}
    assert len(results) == 1, "Sticky session should always return the same model"


def test_sticky_different_ips(router, policy):
    """Different IPs should distribute across both models."""
    router.add_policy(policy)

    models_seen = set()
    for i in range(100):
        chosen = router.resolve("clf", client_ip=f"10.0.0.{i}")
        models_seen.add(chosen)

    # with 100 different IPs and 80/20 split, both models should appear
    assert models_seen == {"clf_v1", "clf_v2"}


def test_auto_rollback(router, policy):
    """When model_b's error rate exceeds threshold, all traffic goes to model_a."""
    policy.sticky = False
    policy.min_requests = 5
    policy.error_threshold = 0.5
    router.add_policy(policy)

    # simulate 5 errors on model_b (100% error rate > 50% threshold)
    for _ in range(5):
        router.record_error("clf_v2")

    # after rollback, all traffic should go to model_a
    for _ in range(50):
        assert router.resolve("clf") == "clf_v1"


def test_no_rollback_under_min_requests(router, policy):
    """Rollback shouldn't trigger with fewer than min_requests."""
    policy.sticky = False
    policy.min_requests = 10
    router.add_policy(policy)

    # only 3 errors (below min_requests=10)
    for _ in range(3):
        router.record_error("clf_v2")

    # model_b should still receive some traffic
    models_seen = set()
    for _ in range(200):
        models_seen.add(router.resolve("clf"))

    assert "clf_v2" in models_seen


def test_policy_crud(router, policy):
    """Add, list, and remove policies."""
    router.add_policy(policy)

    # list
    policies = router.list_policies()
    assert len(policies) == 1
    assert policies[0]["name"] == "clf"
    assert policies[0]["min_requests"] == 5

    # get
    assert router.get_policy("clf") is not None
    assert router.get_policy("nonexistent") is None

    # remove
    assert router.remove_policy("clf") is True
    assert router.remove_policy("clf") is False  # already removed
    assert router.list_policies() == []
