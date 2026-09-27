"""The executable harness patch representation and response parser.

The patch language is intentionally narrow. A Harness-R1 patch is a single JSON
object with top-level ``benchmark``, ``description``, and ``actions``; every
action is ``{"type": "add_code_hook", "hook": ..., "code": ...}`` naming one of
the four lifecycle positions. This is the released ``prefill_think_patch``
protocol (paper Appendix A; reference ``harness_r1_patch.py``).
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any

from nova.core.harness import HookNames
from nova.core.types import to_jsonable
from nova.errors import PatchParseError, PatchValidationError

SCHEMA_VERSION = "harness-r1-patch-v1"

_ACTION_TYPE = "add_code_hook"
_ASSISTANT_TAG = "<|im_start|>assistant"
_THINK_RE = re.compile(r"<think>\s*(.*?)\s*</think>", re.DOTALL | re.IGNORECASE)
_PATCH_RE = re.compile(r"<patch>\s*(.*?)\s*</patch>", re.DOTALL | re.IGNORECASE)
_PATCH_OPEN_RE = re.compile(r"<patch>\s*(.*)", re.DOTALL | re.IGNORECASE)
_FENCE_RE = re.compile(r"```(?:json)?\s*(\{.*?\})\s*```", re.DOTALL)


@dataclass(frozen=True, slots=True)
class CodeHook:
    """One ``add_code_hook`` action."""

    hook: str
    code: str

    def __post_init__(self) -> None:
        if self.hook not in HookNames:
            raise PatchValidationError(f"unsupported code hook: {self.hook!r}")

    def to_dict(self) -> dict[str, Any]:
        return {"type": _ACTION_TYPE, "hook": self.hook, "code": self.code}


@dataclass(frozen=True, slots=True)
class HarnessPatch:
    """A parsed, structurally valid executable harness patch."""

    benchmark: str
    hooks: tuple[CodeHook, ...]
    description: str = ""
    schema_version: str = SCHEMA_VERSION
    think: str = ""
    source: str = "engineer"
    raw: dict[str, Any] = field(default_factory=dict, compare=False, repr=False)

    @property
    def hook_names(self) -> tuple[str, ...]:
        return tuple(hook.hook for hook in self.hooks)

    @property
    def is_empty(self) -> bool:
        return not self.hooks

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "benchmark": self.benchmark,
            "description": self.description,
            "actions": [hook.to_dict() for hook in self.hooks],
        }

    @classmethod
    def from_dict(cls, raw: Any, *, bench: str | None = None, source: str = "engineer") -> HarnessPatch:
        """Build a patch from an already-decoded mapping (no execution)."""
        if not isinstance(raw, dict):
            raise PatchValidationError("patch must be a JSON object")
        benchmark = str(raw.get("benchmark") or bench or "").strip().lower()
        if not benchmark:
            raise PatchValidationError("patch.benchmark is required")
        actions = raw.get("actions")
        if not isinstance(actions, list) or not actions:
            raise PatchValidationError("patch.actions must be a non-empty list")
        hooks: list[CodeHook] = []
        for idx, action in enumerate(actions):
            if not isinstance(action, dict):
                raise PatchValidationError(f"action {idx} must be an object")
            if action.get("type") != _ACTION_TYPE:
                raise PatchValidationError(
                    f"action {idx} type {action.get('type')!r} is not allowed in the code-hook-only protocol"
                )
            hook = str(action.get("hook") or "").strip()
            code = action.get("code")
            if not isinstance(code, str) or not code.strip():
                raise PatchValidationError(f"action {idx} must include non-empty code")
            hooks.append(CodeHook(hook=hook, code=code))
        description = str(raw.get("description") or "").strip()
        return cls(
            benchmark=benchmark,
            hooks=tuple(hooks),
            description=description,
            schema_version=str(raw.get("schema_version") or SCHEMA_VERSION),
            source=source,
            raw=dict(raw),
        )

    def to_json(self) -> str:
        return json.dumps(to_jsonable(self.to_dict()), indent=2, ensure_ascii=False)


# --------------------------------------------------------------------------- #
# Response parsing
# --------------------------------------------------------------------------- #
def extract_json_object(text: str) -> dict[str, Any]:
    """Extract the last plausible JSON object from arbitrary model text."""
    if not isinstance(text, str) or not text.strip():
        raise PatchParseError("no JSON object found in model output")
    text = text.strip()

    def candidates(segment: str) -> list[dict[str, Any]]:
        decoder = json.JSONDecoder()
        objects: list[dict[str, Any]] = []
        for match in re.finditer(r"\{", segment):
            try:
                obj, _ = decoder.raw_decode(segment[match.start() :])
            except json.JSONDecodeError:
                continue
            if isinstance(obj, dict):
                objects.append(obj)
        return objects

    def choose(objects: list[dict[str, Any]]) -> dict[str, Any] | None:
        patch_like = [
            obj for obj in objects if isinstance(obj.get("actions"), list) and isinstance(obj.get("benchmark"), str)
        ]
        if patch_like:
            return patch_like[-1]
        return objects[-1] if objects else None

    search = text
    if _ASSISTANT_TAG in text:
        search = text.rsplit(_ASSISTANT_TAG, 1)[-1]

    segments: list[str] = []
    blocks = [match.group(1) for match in _PATCH_RE.finditer(search)]
    if not blocks:
        match = _PATCH_OPEN_RE.search(search)
        if match:
            blocks.append(match.group(1))
    segments.extend(blocks)
    segments.append(search)
    segments.extend(match.group(1) for match in _FENCE_RE.finditer(search))
    if search != text:
        segments.append(text)

    for segment in segments:
        obj = choose(candidates(segment))
        if obj is not None:
            return obj
    raise PatchParseError("no JSON object found in model output")


def _assistant_segment(text: str) -> str:
    return text.rsplit(_ASSISTANT_TAG, 1)[-1] if _ASSISTANT_TAG in text else text


def extract_think_patch(text: str) -> tuple[str, dict[str, Any]]:
    """Extract ``(think, patch_json)`` from a complete ``<think>``/``<patch>`` response."""
    segment = _assistant_segment(text.strip())
    think_match = _THINK_RE.search(segment)
    if think_match is None:
        raise PatchParseError("missing complete <think>...</think> block")
    patch_matches = list(_PATCH_RE.finditer(segment))
    if not patch_matches:
        raise PatchParseError("missing complete <patch>...</patch> block")
    patch_text = patch_matches[-1].group(1).strip()
    if not patch_text:
        raise PatchParseError("<patch> block is empty")
    return think_match.group(1).strip(), extract_json_object(f"<patch>\n{patch_text}\n</patch>")


def extract_prefilled_think_patch(text: str) -> tuple[str, dict[str, Any]]:
    """Extract from a completion whose prompt already opened ``<think>``."""
    segment = _assistant_segment(text.strip()).lstrip()
    if re.match(r"(?is)^<think\b", segment):
        raise PatchParseError("prefilled-think response must not start with another <think> tag")
    return extract_think_patch("<think>\n" + segment)


def extract_patch(
    text: str,
    *,
    benchmark: str | None = None,
    require_think: bool = False,
    prefill_think: bool = False,
    source: str = "engineer",
) -> HarnessPatch:
    """Parse model output into a :class:`HarnessPatch`.

    Args:
        text: Raw model output.
        benchmark: Expected benchmark id. If given, the parsed patch must match.
        require_think: Require an explicit ``<think>...</think>`` block.
        prefill_think: The prompt opened ``<think>``; the response continues it.
        source: Provenance label for the patch.
    """
    think = ""
    if prefill_think:
        think, raw = extract_prefilled_think_patch(text)
    elif require_think:
        think, raw = extract_think_patch(text)
    else:
        raw = extract_json_object(text)
    patch = HarnessPatch.from_dict(raw, bench=benchmark, source=source)
    if benchmark is not None and patch.benchmark != benchmark.strip().lower():
        raise PatchValidationError(f"patch benchmark {patch.benchmark!r} does not match expected {benchmark!r}")
    if think:
        patch = HarnessPatch(
            benchmark=patch.benchmark,
            hooks=patch.hooks,
            description=patch.description,
            schema_version=patch.schema_version,
            think=think,
            source=patch.source,
            raw=patch.raw,
        )
    return patch
