"""Random safe-hook engineer (used by the lifecycle/ablation studies).

This engineer samples small, *safe*, generic hook bodies from a fixed template
library. It is not a competitor to a trained engineer; it exists to isolate how
much of the gain comes from outcome-grounded *selection* versus the content of
the proposal (ablation B: random harness patches).
"""

from __future__ import annotations

import random
from collections.abc import Sequence
from typing import Any

from nova.engineering.harness_engineer import BaseHarnessEngineer
from nova.engineering.patch import CodeHook, HarnessPatch

_TEMPLATES: tuple[tuple[str, str], ...] = (
    (
        "on_init",
        "def hook(ctx, nb):\n"
        "    nb['stage'] = nb.get('stage', 0)\n"
        "    return {'skills': [{'text': 'Read the latest observation carefully before acting.'}]}\n",
    ),
    (
        "make_pre_hint",
        "def hook(ctx, nb):\n"
        "    obs = str(ctx.get('observation', '')).lower()\n"
        "    if 'error' in obs or 'fail' in obs:\n"
        "        return {'message': 'The last step failed; inspect it before retrying.'}\n"
        "    return None\n",
    ),
    (
        "on_before_action",
        "def hook(ctx, nb):\n"
        "    state = ctx.get('state') or {}\n"
        "    if state.get('repeated_action_count', 0) >= 2:\n"
        "        return {'kind': 'block_and_prompt', 'message': 'This action repeats; choose a different one.'}\n"
        "    return None\n",
    ),
    (
        "on_before_action",
        "def hook(ctx, nb):\n"
        "    state = ctx.get('state') or {}\n"
        "    if state.get('remaining_steps', 99) <= 2:\n"
        "        return {'kind': 'block_and_prompt', 'message': 'Few steps remain; act decisively.'}\n"
        "    return None\n",
    ),
    (
        "on_post_step",
        "def hook(ctx, nb):\n"
        "    obs = str(ctx.get('observation', '')).lower()\n"
        "    if 'nothing happens' in obs:\n"
        "        return {'kind': 'inject_hint', 'message': 'That action had no effect; change strategy.'}\n"
        "    return None\n",
    ),
    (
        "on_post_step",
        "def hook(ctx, nb):\n"
        "    nb['steps'] = nb.get('steps', 0) + 1\n"
        "    if nb['steps'] % 4 == 0:\n"
        "        return {'kind': 'inject_hint', 'message': 'Take stock of progress toward the task goal.'}\n"
        "    return None\n",
    ),
)


class RandomHarnessEngineer(BaseHarnessEngineer):
    """Sample deterministic random safe-hook patches."""

    name = "random-engineer"

    def __init__(
        self,
        *,
        benchmark: str,
        seed: int = 0,
        templates: Sequence[tuple[str, str]] | None = None,
        max_hooks: int = 2,
    ) -> None:
        self.benchmark = benchmark
        self.seed = seed
        self.templates = list(templates or _TEMPLATES)
        self.max_hooks = max(1, max_hooks)
        self._rng = random.Random(seed)
        self._counter = 0

    def generate_patch(self, packet: Any) -> HarnessPatch:
        self._counter += 1
        count = self._rng.randint(1, self.max_hooks)
        chosen = self._rng.sample(self.templates, k=min(count, len(self.templates)))
        # Keep at most one hook per lifecycle position.
        seen: set[str] = set()
        hooks: list[CodeHook] = []
        for hook_name, code in chosen:
            if hook_name in seen:
                continue
            seen.add(hook_name)
            hooks.append(CodeHook(hook=hook_name, code=code))
        return HarnessPatch(
            benchmark=self.benchmark,
            description=f"random safe-hook patch #{self._counter}",
            hooks=tuple(hooks),
            source="random-engineer",
        )

    def generate_candidates(self, packet: Any, n: int) -> list[HarnessPatch]:
        return [self.generate_patch(packet) for _ in range(max(0, n))]
