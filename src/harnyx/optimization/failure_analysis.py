"""Failure extraction and compact failure packets.

A failure packet contains the *minimum* evidence the harness engineer needs:
the failed episodes only, their task constraints and action/observation excerpts,
recurring runtime signals, and the baseline outcome. It intentionally does not
dump every trajectory or ask a debugger model to label failure modes — the
engineer infers recurring opportunities directly from the traces (reference
``harness_r1_trace_packet.py``).
"""

from __future__ import annotations

import re
from collections import Counter
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable

from harnyx.core.result import EvaluationResult
from harnyx.core.trajectory import Trajectory, TrajectoryStep
from harnyx.core.types import canonical_json

PRODUCT_ID_RE = re.compile(r"\b[bB]0[0-9A-Za-z]{8}\b")
PATH_RE = re.compile(r"/mnt/[^\s`]+")
PRIVATE_PATH_RE = re.compile(r"/(?:root|home|tmp)/[^\s`'\"),;]+")

DEFAULT_MAX_PROMPT_CHARS = 72_000


def sanitize(text: str) -> str:
    """Redact identifiers and private paths from trace text."""
    text = PRODUCT_ID_RE.sub("[PRODUCT_ID]", text)
    text = PATH_RE.sub("[PATH]", text)
    text = PRIVATE_PATH_RE.sub("[PATH]", text)
    return text


def trim(text: str, max_chars: int) -> str:
    """Trim ``text`` to ``max_chars``, keeping both ends with a middle marker."""
    text = sanitize(str(text or "")).strip()
    text = re.sub(r"\n{3,}", "\n\n", text)
    if len(text) <= max_chars:
        return text
    head = text[: max_chars // 2].rstrip()
    tail = text[-max_chars // 2 :].lstrip()
    return head + "\n...[middle truncated]...\n" + tail


@dataclass(slots=True)
class FailureCase:
    """One failed trajectory compacted into evidence."""

    task_id: str
    reward: float
    success: bool = False
    status: str = "unknown"
    steps: list[TrajectoryStep] = field(default_factory=list)
    signals: dict[str, Any] = field(default_factory=dict)
    instruction: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "task_id": self.task_id,
            "reward": self.reward,
            "success": self.success,
            "status": self.status,
            "signals": self.signals,
            "instruction": self.instruction,
            "steps": [step.to_dict() for step in self.steps],
        }

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> FailureCase:
        steps = [
            TrajectoryStep(
                index=int(step.get("index", i)),
                observation=str(step.get("observation", "")),
                action=step.get("action"),
                tool_result=step.get("tool_result"),
                agent_output=step.get("agent_output"),
                harness_effects=list(step.get("harness_effects", []) or []),
                error=step.get("error"),
                timestamp=float(step.get("timestamp", 0.0)),
                state=dict(step.get("state", {}) or {}),
            )
            for i, step in enumerate(raw.get("steps", []) or [])
        ]
        return cls(
            task_id=str(raw.get("task_id", "")),
            reward=float(raw.get("reward", 0.0)),
            success=bool(raw.get("success", False)),
            status=str(raw.get("status", "unknown")),
            steps=steps,
            signals=dict(raw.get("signals", {}) or {}),
            instruction=str(raw.get("instruction", "")),
        )


@dataclass(slots=True)
class FailurePacket:
    """A batch-conditioned evidence bundle for the harness engineer."""

    benchmark: str
    batch_id: str = "batch"
    cases: list[FailureCase] = field(default_factory=list)
    baseline_rewards: dict[str, float] = field(default_factory=dict)
    baseline_successes: dict[str, bool] = field(default_factory=dict)
    signals: dict[str, Any] = field(default_factory=dict)
    selection: str = "round_robin"
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def failed_task_ids(self) -> list[str]:
        return [case.task_id for case in self.cases]

    @property
    def baseline_pass(self) -> int:
        return sum(1 for value in self.baseline_successes.values() if value)

    @property
    def batch_size(self) -> int:
        return len(self.baseline_rewards)

    @property
    def baseline_mean_reward(self) -> float:
        if not self.baseline_rewards:
            return 0.0
        return sum(self.baseline_rewards.values()) / len(self.baseline_rewards)

    def to_dict(self) -> dict[str, Any]:
        return {
            "benchmark": self.benchmark,
            "batch_id": self.batch_id,
            "cases": [case.to_dict() for case in self.cases],
            "baseline_rewards": self.baseline_rewards,
            "baseline_successes": self.baseline_successes,
            "signals": self.signals,
            "selection": self.selection,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> FailurePacket:
        return cls(
            benchmark=str(raw.get("benchmark", "")),
            batch_id=str(raw.get("batch_id", "batch")),
            cases=[FailureCase.from_dict(item) for item in raw.get("cases", []) or []],
            baseline_rewards={str(k): float(v) for k, v in (raw.get("baseline_rewards") or {}).items()},
            baseline_successes={str(k): bool(v) for k, v in (raw.get("baseline_successes") or {}).items()},
            signals=dict(raw.get("signals", {}) or {}),
            selection=str(raw.get("selection", "round_robin")),
            metadata=dict(raw.get("metadata", {}) or {}),
        )

    def fingerprint(self) -> str:
        """Deterministic content hash of the packet (for caching/dedup)."""
        import hashlib

        payload = {
            "benchmark": self.benchmark,
            "cases": [case.task_id for case in self.cases],
            "rewards": self.baseline_rewards,
        }
        return hashlib.sha256(canonical_json(payload).encode()).hexdigest()[:16]

    def to_prompt_text(
        self,
        *,
        max_obs_chars: int = 1200,
        max_assistant_chars: int = 600,
        max_prompt_chars: int = DEFAULT_MAX_PROMPT_CHARS,
    ) -> str:
        """Render the packet as the deterministic prompt text."""
        case_list = self.cases
        lines = [
            "# Harness-R1 Direct Failure Trace Packet",
            "",
            "These are deterministic extracts from no-harness rollout failures.",
            "No debugger model has labeled the failure modes. Infer recurring "
            "harness-edit opportunities from the traces.",
            "",
            "## Scope",
            "",
            f"- Benchmark: `{self.benchmark}`",
            f"- Batch: `{self.batch_id}`",
            f"- Failure traces included: `{len(case_list)}`",
            f"- Baseline fully-successful tasks: `{self.baseline_pass}/{self.batch_size}`",
            f"- Baseline mean reward: `{self.baseline_mean_reward:.4f}`",
            f"- Selection strategy: `{self.selection}`",
            "",
            "## Instructions for Harness Engineer",
            "",
            "- Infer recurring, general failure modes from the traces before writing a patch.",
            "- Do not encode product ids, exact object/location instance ids, task indices, or benchmark answers.",
            "- Use hard blocking only when the runtime condition is clearly observable; otherwise prefer short hints.",
            "",
            "## Failure Traces",
            "",
        ]
        used = sum(len(line) for line in lines)
        included = 0
        for case in case_list:
            block = render_case(case, benchmark=self.benchmark, max_obs_chars=max_obs_chars, max_assistant_chars=max_assistant_chars)
            if included and used + len(block) > max_prompt_chars:
                break
            lines.append(block)
            lines.append("")
            used += len(block)
            included += 1
        lines.append(f"Included traces after budget: `{included}`")
        return "\n".join(lines).rstrip() + "\n"


def render_case(
    case: FailureCase,
    *,
    benchmark: str,
    max_obs_chars: int,
    max_assistant_chars: int,
) -> str:
    """Render one failure case as a compact, sanitized trace."""
    lines = [
        "## Trace case",
        "",
        f"- task_id: `{sanitize(case.task_id)}`",
        f"- final_reward: `{case.reward:.4f}`",
        f"- status: `{case.status}`",
        f"- turns: `{len(case.steps)}`",
        "",
        "Runtime signals:",
    ]
    for key in sorted(case.signals):
        lines.append(f"- {key}: `{case.signals[key]}`")
    lines.append("")
    if case.instruction:
        lines.append("TASK:")
        lines.append(trim(case.instruction, max_obs_chars))
        lines.append("")
    for step in case.steps:
        if step.action is not None:
            name = step.action.get("name", "")
            value = step.action.get("raw") or step.action.get("arguments", {}).get("value", "")
            lines.append(f"ASSISTANT_TOOL: {name} {sanitize(str(value))}".rstrip())
        elif step.agent_output:
            lines.append("ASSISTANT_TEXT:")
            lines.append(trim(step.agent_output, max_assistant_chars))
        if step.tool_result:
            lines.append("OBSERVATION:")
            lines.append(trim(step.tool_result, max_obs_chars))
        for effect in step.harness_effects:
            lines.append(f"HARNESS_EFFECT: {sanitize(canonical_json(effect))}")
        if step.error:
            lines.append(f"ERROR: {sanitize(step.error)}")
        lines.append("")
    return "\n".join(lines).rstrip()


def compute_signals(trajectory: Trajectory) -> dict[str, Any]:
    """Compute deterministic runtime signals from a trajectory.

    These mirror the reference ``runtime_signal_lines`` counters but are derived
    from Harnyx's structured steps, so they are benchmark-agnostic. The
    ``alfworld`` no-op/repeat signals are computed from observation text and are
    harmless for other benchmarks.
    """
    actions = trajectory.actions()
    action_names = [str(action.get("name", "")) for action in actions]
    action_values = [f"{action.get('name','')}:{str(action.get('raw') or action.get('arguments', {}).get('value','')).lower()}" for action in actions]
    repeated = {key: count for key, count in Counter(action_values).items() if count > 1}
    name_counts = Counter(action_names)

    observations = [step.tool_result or step.observation for step in trajectory.steps]
    norm_obs = [re.sub(r"\s+", " ", sanitize(obs).strip().lower())[:200] for obs in observations if obs]
    repeated_obs = {key: count for key, count in Counter(norm_obs).items() if count > 1 and key}
    no_op = sum(1 for obs in observations if "nothing happens" in (obs or "").lower())
    errors = [step.error for step in trajectory.steps if step.error]

    signals: dict[str, Any] = {
        "tool_actions": len(actions),
        "action_name_counts": dict(sorted(name_counts.items())),
        "repeated_action_values": len(repeated),
        "no_op_observations": no_op,
        "repeated_observation_groups": len(repeated_obs),
        "errors": len(errors),
        "harness_interventions": len(trajectory.harness_effects()),
    }
    if trajectory.status:
        signals["terminal_status"] = trajectory.status
    return signals


def select_cases(cases: Sequence[FailureCase], max_traces: int, strategy: str) -> list[FailureCase]:
    """Select which failures enter the packet (mirrors reference strategies)."""
    if max_traces <= 0 or len(cases) <= max_traces:
        return list(cases)
    if strategy == "lowest_reward":
        return sorted(cases, key=lambda case: (case.reward, case.task_id))[:max_traces]
    if strategy == "first":
        return list(cases[:max_traces])
    if strategy == "round_robin":
        grouped: dict[str, list[FailureCase]] = {}
        for case in sorted(cases, key=lambda case: (case.task_id, case.reward)):
            grouped.setdefault(case.task_id.split(":")[0], []).append(case)
        out: list[FailureCase] = []
        while len(out) < max_traces and any(grouped.values()):
            for key in sorted(grouped):
                bucket = grouped[key]
                if bucket:
                    out.append(bucket.pop(0))
                    if len(out) >= max_traces:
                        break
        return out
    raise ValueError(f"unknown selection strategy: {strategy}")


@runtime_checkable
class FailureAnalyzer(Protocol):
    """Structural interface for extracting a failure packet from trajectories."""

    def analyze(
        self,
        trajectories: Iterable[Trajectory],
        *,
        baseline: EvaluationResult | None = None,
        batch_id: str = "batch",
    ) -> FailurePacket: ...


class TraceFailureAnalyzer:
    """Default deterministic failure-packet builder."""

    def __init__(
        self,
        *,
        benchmark: str = "",
        reward_threshold: float = 1.0,
        max_traces: int = 20,
        strategy: str = "round_robin",
    ) -> None:
        self.benchmark = benchmark
        self.reward_threshold = reward_threshold
        self.max_traces = max_traces
        self.strategy = strategy

    def analyze(
        self,
        trajectories: Iterable[Trajectory],
        *,
        baseline: EvaluationResult | None = None,
        batch_id: str = "batch",
    ) -> FailurePacket:
        all_trajectories = list(trajectories)
        rewards: dict[str, float] = {}
        successes: dict[str, bool] = {}
        if baseline is not None:
            rewards.update(baseline.rewards)
            successes.update(baseline.successes)
        for trajectory in all_trajectories:
            rewards.setdefault(trajectory.task_id, trajectory.final_reward)
            successes.setdefault(trajectory.task_id, trajectory.success)

        cases: list[FailureCase] = []
        for trajectory in all_trajectories:
            if trajectory.success or trajectory.final_reward >= self.reward_threshold:
                continue
            cases.append(
                FailureCase(
                    task_id=trajectory.task_id,
                    reward=trajectory.final_reward,
                    success=trajectory.success,
                    status=trajectory.status,
                    steps=list(trajectory.steps),
                    signals=compute_signals(trajectory),
                    instruction=str(trajectory.metadata.get("instruction", "")),
                )
            )

        selected = select_cases(cases, self.max_traces, self.strategy)
        aggregate = self._aggregate_signals(selected)
        return FailurePacket(
            benchmark=self.benchmark,
            batch_id=batch_id,
            cases=selected,
            baseline_rewards=rewards,
            baseline_successes=successes,
            signals=aggregate,
            selection=self.strategy,
            metadata={
                "candidate_failures": len(cases),
                "included": len(selected),
            },
        )

    @staticmethod
    def _aggregate_signals(cases: Sequence[FailureCase]) -> dict[str, Any]:
        totals: Counter[str] = Counter()
        for case in cases:
            for key in (
                "tool_actions",
                "repeated_action_values",
                "no_op_observations",
                "errors",
                "harness_interventions",
            ):
                value = case.signals.get(key)
                if isinstance(value, (int, float)):
                    totals[key] += int(value)
        return {"cases": len(cases), **{key: totals[key] for key in sorted(totals)}}
