"""Online GRPO interface for the harness engineer.

Reference configuration (paper Appendix B.2, released
``configs/rl/mixed_codepatch.yaml`` and ``scripts/train_engineer_rl.sh``):

    K = 8 candidates per prompt, 4 prompts per rollout batch, global batch 32,
    4 rollout iterations per update, max policy staleness 8, lr 1e-6 (constant),
    Adam (0.9, 0.98), weight decay 0.1, GRPO clip 0.20/0.28, truncated
    importance sampling with max weight 2.0, no entropy/KL bonus, rollout
    temperature/top-p 0.7/0.95, max prompt/response 28672/12288.

The reward is the realized same-batch performance change from rerunning the
frozen target — never a learned judge. :class:`HarnessPatchReward` bridges the
NOVA reward into a TRL-compatible reward callable. The reference implementation
uses the authors' Relax framework; NOVA exposes the identical hyperparameters
and a TRL-based launcher rather than vendoring Relax.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from nova.core.types import to_jsonable, write_json
from nova.engineering.patch import HarnessPatch, extract_patch
from nova.engineering.validation import PatchValidator
from nova.errors import ConfigError
from nova.sandbox.isolation import SubprocessSandbox

EvaluatePatch = Callable[[HarnessPatch | None, Mapping[str, Any]], float]


@dataclass(slots=True)
class GRPOConfig:
    """Online GRPO hyperparameters (paper Appendix B.2)."""

    base_model: str = "Qwen3.5-9B"
    dataset_path: str = "data/engineer_rl.jsonl"
    output_dir: str = "outputs/engineer-grpo"
    num_generations: int = 8
    rollout_batch_size: int = 4
    global_batch_size: int = 32
    rollout_iterations: int = 4
    max_policy_staleness: int = 8
    learning_rate: float = 1e-6
    lr_scheduler: str = "constant"
    optimizer: str = "adam"
    adam_beta1: float = 0.9
    adam_beta2: float = 0.98
    weight_decay: float = 0.1
    clip_lower: float = 0.20
    clip_upper: float = 0.28
    truncated_importance_sampling: bool = True
    importance_sampling_max_weight: float = 2.0
    entropy_coefficient: float = 0.0
    kl_coefficient: float = 0.0
    temperature: float = 0.7
    top_p: float = 0.95
    max_prompt_length: int = 28672
    max_response_length: int = 12288
    train_seed: int = 1234
    rollout_seed: int = 42

    def to_dict(self) -> dict[str, Any]:
        return to_jsonable(asdict(self))


class HarnessPatchReward:
    """TRL-compatible reward callable backed by NOVA outcome evaluation.

    Args:
        evaluate_patch: Callable ``(patch_or_none, sample_metadata) -> reward``.
            It should parse/validate the patch and rerun the frozen target on the
            same task batch, returning ``0.0`` for invalid/incomplete cases.
        validator: Static validator; invalid completions score zero without
            touching the sandbox.
        smoke_sandbox: Optional process-isolated smoke test performed before
            evaluation.
    """

    def __init__(
        self,
        evaluate_patch: EvaluatePatch,
        *,
        validator: PatchValidator | None = None,
        smoke_sandbox: SubprocessSandbox | None = None,
    ) -> None:
        self.evaluate_patch = evaluate_patch
        self.validator = validator or PatchValidator(smoke_sandbox=smoke_sandbox)

    def __call__(
        self,
        completions: Sequence[str],
        prompts: Sequence[Any] | None = None,
        benchmark: Sequence[str] | str | None = None,
        **kwargs: Any,
    ) -> list[float]:
        rewards: list[float] = []
        benchmarks = _align(benchmark, len(completions))
        for completion, bench in zip(completions, benchmarks, strict=False):
            patch: HarnessPatch | None = None
            try:
                candidate = extract_patch(completion, benchmark=bench or None, prefill_think=True)
                result = self.validator.validate(candidate, benchmark=bench or None)
                if result.ok:
                    patch = result.patch
            except Exception:
                patch = None
            metadata: dict[str, Any] = {"benchmark": bench}
            metadata.update({key: value for key, value in kwargs.items() if _is_alignable(value, len(completions))})
            rewards.append(float(self.evaluate_patch(patch, metadata)))
        return rewards


def _align(value: Sequence[str] | str | None, n: int) -> list[str]:
    if value is None:
        return [""] * n
    if isinstance(value, str):
        return [value] * n
    items = [str(item) for item in value]
    if len(items) < n:
        items = items + [""] * (n - len(items))
    return items[:n]


def _is_alignable(value: Any, n: int) -> bool:
    try:
        return len(value) == n  # type: ignore[arg-type]
    except TypeError:
        return False


def train_grpo(config: GRPOConfig, *, reward_fn: Callable[..., list[float]], dataset_path: str | None = None) -> Any:
    """Launch online GRPO with ``trl.GRPOTrainer``.

    Raises:
        ConfigError: when ``trl`` is not installed, with an explicit message.
    """
    try:
        from datasets import load_dataset  # type: ignore[import-untyped]
        from trl import GRPOConfig as TRLGRPOConfig
        from trl import GRPOTrainer  # type: ignore[import-untyped]
    except ImportError as exc:  # pragma: no cover - depends on environment
        raise ConfigError(
            "GRPO requires the optional 'train' extra (trl, transformers, datasets, torch). "
            "Install with `pip install nova-harness[train]`, or use the reference Relax "
            "stack via configs/rl/mixed_codepatch.yaml."
        ) from exc

    dataset = load_dataset("json", data_files=dataset_path or config.dataset_path, split="train")
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    write_json(output_dir / "grpo_config.json", config.to_dict())

    training_args = TRLGRPOConfig(
        output_dir=str(output_dir),
        num_generations=config.num_generations,
        per_device_train_batch_size=max(1, config.global_batch_size // config.num_generations),
        gradient_accumulation_steps=max(1, config.num_generations),
        learning_rate=config.learning_rate,
        lr_scheduler_type=config.lr_scheduler,
        optim=config.optimizer,
        adam_beta1=config.adam_beta1,
        adam_beta2=config.adam_beta2,
        weight_decay=config.weight_decay,
        temperature=config.temperature,
        top_p=config.top_p,
        max_prompt_length=config.max_prompt_length,
        max_completion_length=config.max_response_length,
        beta=config.kl_coefficient,
        epsilon=config.clip_lower,
        epsilon_high=config.clip_upper,
        seed=config.train_seed,
        data_seed=config.rollout_seed,
        logging_steps=10,
        report_to=[],
    )
    trainer = GRPOTrainer(model=config.base_model, args=training_args, train_dataset=dataset, reward_funcs=[reward_fn])
    trainer.train()
    trainer.save_model(str(output_dir / "final"))
    return trainer
