"""Task definition: the unit of work an agent is evaluated on."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

from nova.core.types import canonical_json, to_jsonable


@dataclass(frozen=True, slots=True)
class Task:
    """A single benchmark-agnostic task.

    Args:
        id: Stable identity used to pair baseline and patched reruns. The
            Harness-R1 reward is only meaningful when the *same* task ids are
            executed before and after a patch.
        instruction: Natural-language instruction shown to the agent.
        payload: Benchmark-specific structured data (e.g. goal metadata, SQL
            schema). Never assumed by core code.
        metadata: Bookkeeping data (family, split, tags).
    """

    id: str
    instruction: str
    payload: Mapping[str, Any] = field(default_factory=dict)
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return to_jsonable(self)

    def key(self) -> str:
        """Return a deterministic content hash used for regression suites."""
        import hashlib

        return hashlib.sha256(canonical_json({"id": self.id, "instruction": self.instruction}).encode()).hexdigest()[:16]
