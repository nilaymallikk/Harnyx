"""Training support: SFT dataset construction and a GRPO interface.

Training is kept out of the core runtime. Users can use a pretrained engineer
without importing anything here.
"""

from __future__ import annotations

from harnyx.training.dataset import (
    SFTDatasetBuilder,
    SFTExample,
    filter_training_records,
    write_sft_dataset,
)
from harnyx.training.grpo import GRPOConfig, HarnessPatchReward, train_grpo
from harnyx.training.sft import SFTConfig, train_sft

__all__ = [
    "GRPOConfig",
    "HarnessPatchReward",
    "SFTConfig",
    "SFTDatasetBuilder",
    "SFTExample",
    "filter_training_records",
    "train_grpo",
    "train_sft",
    "write_sft_dataset",
]
