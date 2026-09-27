"""Small, deterministic metric helpers."""

from __future__ import annotations

from collections.abc import Mapping, Sequence


def mean_reward(rewards: Mapping[str, float] | Sequence[float]) -> float:
    """Mean of a reward mapping or sequence (0.0 when empty)."""
    values = list(rewards.values()) if isinstance(rewards, Mapping) else list(rewards)
    return sum(float(v) for v in values) / len(values) if values else 0.0


def success_rate(successes: Mapping[str, bool] | Sequence[bool]) -> float:
    """Fraction of successful tasks (0.0 when empty)."""
    values = list(successes.values()) if isinstance(successes, Mapping) else list(successes)
    return sum(1 for v in values if v) / len(values) if values else 0.0


def aggregate_rewards(
    rewards: Mapping[str, float],
    successes: Mapping[str, bool],
) -> dict[str, float | int]:
    """Return the summary metrics reported by NOVA runs."""
    return {
        "n": len(rewards),
        "mean_reward": mean_reward(rewards),
        "success_rate": success_rate(successes),
        "num_success": sum(1 for value in successes.values() if value),
    }
