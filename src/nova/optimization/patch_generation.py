"""Candidate patch generation.

One failure packet yields N candidate patches (paper reference: K = 8). NOVA
generates as many as the engineer returns, validates each against the patch
contract, and de-duplicates identical proposals so the same patch is not
evaluated twice.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

from nova.core.types import canonical_json, to_jsonable
from nova.engineering.harness_engineer import HarnessEngineer
from nova.engineering.patch import HarnessPatch
from nova.engineering.validation import PatchValidator, ValidationResult
from nova.errors import EngineerError

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class CandidatePatch:
    """One generated patch plus its static validation outcome."""

    index: int
    patch: HarnessPatch | None
    validation: ValidationResult
    error: str | None = None
    duplicate_of: int | None = None

    @property
    def is_valid(self) -> bool:
        return self.validation.ok and self.patch is not None

    def fingerprint(self) -> str:
        if self.patch is None:
            return f"error:{self.error}"
        return canonical_json(self.patch.to_dict())

    def to_dict(self) -> dict[str, Any]:
        data = {
            "index": self.index,
            "valid": self.is_valid,
            "errors": list(self.validation.errors),
            "warnings": list(self.validation.warnings),
            "duplicate_of": self.duplicate_of,
            "error": self.error,
            "patch": self.patch.to_dict() if self.patch is not None else None,
        }
        return to_jsonable(data)


class PatchGenerator:
    """Generate and statically validate candidate patches for a packet."""

    def __init__(
        self,
        engineer: HarnessEngineer,
        validator: PatchValidator | None = None,
        *,
        deduplicate: bool = True,
    ) -> None:
        self.engineer = engineer
        self.validator = validator or PatchValidator()
        self.deduplicate = deduplicate

    @property
    def name(self) -> str:
        return getattr(self.engineer, "name", "engineer")

    def generate(self, packet: Any, n: int) -> list[CandidatePatch]:
        """Return up to ``n`` candidate patches, each statically validated."""
        if n < 1:
            raise ValueError("n must be >= 1")
        try:
            patches = self.engineer.generate_candidates(packet, n)
        except EngineerError as exc:
            logger.warning("engineer failed for packet %s: %s", getattr(packet, "batch_id", "?"), exc)
            return [_failed_candidate(0, str(exc))]

        candidates: list[CandidatePatch] = []
        seen: dict[str, int] = {}
        for index, patch in enumerate(patches):
            result = self.validator.validate(patch)
            duplicate_of = None
            if self.deduplicate and result.ok and patch is not None:
                fp = canonical_json(patch.to_dict())
                if fp in seen:
                    duplicate_of = seen[fp]
                else:
                    seen[fp] = index
            candidates.append(CandidatePatch(index=index, patch=result.patch, validation=result, duplicate_of=duplicate_of))
        return candidates


def _failed_candidate(index: int, error: str) -> CandidatePatch:
    return CandidatePatch(
        index=index,
        patch=None,
        validation=ValidationResult(ok=False, errors=[error]),
        error=error,
    )
