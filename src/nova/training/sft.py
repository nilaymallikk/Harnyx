"""Cold-start supervised fine-tuning configuration and launcher.

Reference configuration (paper Appendix B.1, released ``qwen35_9b_engineer_coldstart.yaml``):
full-parameter SFT of a Qwen3.5-9B engineer, 877 examples, 2 epochs, context
32768, global batch 24, AdamW, lr 1e-5, cosine schedule, 3% warmup, bf16, seed 42.

NOVA writes the dataset and exposes the exact hyperparameters. The actual update
step delegates to an installed training framework (``trl``/``transformers``);
if those are not installed, :func:`train_sft` raises an explicit error rather
than pretending to train.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from nova.core.types import to_jsonable, write_json
from nova.errors import ConfigError


@dataclass(slots=True)
class SFTConfig:
    """Cold-start SFT hyperparameters."""

    base_model: str = "Qwen3.5-9B"
    dataset_path: str = "data/engineer_sft.jsonl"
    output_dir: str = "outputs/engineer-sft"
    epochs: int = 2
    context_length: int = 32768
    global_batch_size: int = 24
    learning_rate: float = 1e-5
    lr_scheduler: str = "cosine"
    warmup_ratio: float = 0.03
    optimizer: str = "adamw"
    precision: str = "bf16"
    seed: int = 42
    gradient_checkpointing: bool = True
    num_examples: int | None = None

    def to_dict(self) -> dict[str, Any]:
        return to_jsonable(asdict(self))


def train_sft(config: SFTConfig) -> Any:
    """Launch cold-start SFT with ``trl.SFTTrainer``.

    Raises:
        ConfigError: when ``trl``/``datasets`` are not installed, with an explicit
            message instead of a silent no-op.
    """
    try:
        from datasets import load_dataset  # type: ignore[import-untyped]
        from trl import SFTConfig as TRLSFTConfig
        from trl import SFTTrainer  # type: ignore[import-untyped]
    except ImportError as exc:  # pragma: no cover - depends on environment
        raise ConfigError(
            "SFT requires the optional 'train' extra (trl, transformers, datasets, torch). "
            "Install with `pip install nova-harness[train]`, or use the released reference "
            "configs/sft/qwen35_9b_engineer_coldstart.yaml with LLaMA-Factory."
        ) from exc

    dataset = load_dataset("json", data_files=config.dataset_path, split="train")
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    write_json(output_dir / "sft_config.json", config.to_dict())

    training_args = TRLSFTConfig(
        output_dir=str(output_dir),
        num_train_epochs=config.epochs,
        max_length=config.context_length,
        per_device_train_batch_size=max(1, config.global_batch_size),
        gradient_accumulation_steps=1,
        learning_rate=config.learning_rate,
        lr_scheduler_type=config.lr_scheduler,
        warmup_ratio=config.warmup_ratio,
        optim=config.optimizer,
        bf16=config.precision == "bf16",
        gradient_checkpointing=config.gradient_checkpointing,
        seed=config.seed,
        save_strategy="epoch",
        logging_steps=10,
        report_to=[],
    )
    trainer = SFTTrainer(
        model=config.base_model,
        args=training_args,
        train_dataset=dataset,
    )
    trainer.train()
    trainer.save_model(str(output_dir / "final"))
    return trainer
