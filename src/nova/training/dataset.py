"""Cold-start SFT dataset construction.

Each example is one ordered ``(system, user, assistant)`` triple. The system and
user messages come from the static engineer prompt; the assistant target is the
engineer response: a ``<think>`` block followed by exactly one ``<patch>`` JSON
object. This matches the released prompt protocol so SFT and RL agree byte for
byte.

Teacher filtering in the paper keeps only responses that are executable, complete
the same-batch rerun, and achieve a non-negative reward change. NOVA performs
that filtering at the optimization layer; :func:`filter_training_records` applies
it given recorded rewards.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from nova.core.types import to_jsonable, write_jsonl
from nova.engineering.patch import HarnessPatch
from nova.engineering.prompt import build_engineer_messages
from nova.optimization.failure_analysis import FailurePacket


@dataclass(slots=True)
class SFTExample:
    """One supervised editing example."""

    system: str
    user: str
    assistant: str
    benchmark: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_messages(self) -> list[dict[str, str]]:
        return [
            {"role": "system", "content": self.system},
            {"role": "user", "content": self.user},
            {"role": "assistant", "content": self.assistant},
        ]

    def to_dict(self) -> dict[str, Any]:
        data = to_jsonable(self)
        data["messages"] = self.to_messages()
        return data


def assistant_response_from_patch(patch: HarnessPatch) -> str:
    """Reconstruct the engineer target text for a patch."""
    think = patch.think.strip() or "Derived from recurring observable failures in the packet."
    render = patch.to_dict()
    render.pop("schema_version", None)
    import json

    body = json.dumps(render, indent=2, ensure_ascii=False)
    return f"<think>\n{think}\n</think>\n<patch>\n{body}\n</patch>"


class SFTDatasetBuilder:
    """Build SFT examples from ``(failure_packet, patch)`` pairs."""

    def __init__(self, *, include_response_template: bool = False) -> None:
        self.include_response_template = include_response_template

    def build(self, records: Iterable[tuple[FailurePacket, HarnessPatch]]) -> list[SFTExample]:
        examples: list[SFTExample] = []
        for packet, patch in records:
            messages = build_engineer_messages(
                packet,
                benchmark=packet.benchmark,
                include_response_template=self.include_response_template,
            )
            system = next((m["content"] for m in messages if m["role"] == "system"), "")
            user = "\n\n".join(m["content"] for m in messages if m["role"] != "system")
            examples.append(
                SFTExample(
                    system=system,
                    user=user,
                    assistant=assistant_response_from_patch(patch),
                    benchmark=packet.benchmark,
                    metadata={
                        "packet_fingerprint": packet.fingerprint(),
                        "num_failures": len(packet.cases),
                        "hooks": list(patch.hook_names),
                    },
                )
            )
        return examples


def filter_training_records(
    records: Iterable[tuple[FailurePacket, HarnessPatch, dict[str, Any]]],
    *,
    min_reward: float = 0.0,
    require_valid: bool = True,
) -> list[tuple[FailurePacket, HarnessPatch]]:
    """Keep executable, complete, non-negative-reward teacher edits.

    Each input record is ``(packet, patch, reward_record)`` where ``reward_record``
    has ``valid`` (bool) and ``reward`` (float) keys as produced by the
    optimizer's reward stage.
    """
    kept: list[tuple[FailurePacket, HarnessPatch]] = []
    for packet, patch, reward_record in records:
        if patch is None or patch.is_empty:
            continue
        if require_valid and not bool(reward_record.get("valid", False)):
            continue
        if float(reward_record.get("reward", 0.0)) < min_reward:
            continue
        kept.append((packet, patch))
    return kept


def write_sft_dataset(examples: Iterable[SFTExample], path: str | Path) -> Path:
    """Write SFT examples as JSONL with a ``messages`` field per row."""
    return write_jsonl(path, [example.to_dict() for example in examples])
