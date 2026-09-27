# Training

Training is deliberately separated from the core runtime. You can use a
pretrained engineer without importing `harnyx.training`.

The paper trains in two stages while the target agent stays frozen:

```text
cold-start SFT  ──►  online GRPO  ──►  trained harness engineer
```

## Data

Harnyx builds SFT examples from `(failure_packet, patch)` pairs. Each row is an
ordered `system, user, assistant` triple; the assistant target is a `<think>`
block followed by exactly one `<patch>` JSON object — matching the released
`prefill_think_patch` protocol.

```python
from harnyx.training.dataset import SFTDatasetBuilder, write_sft_dataset

examples = SFTDatasetBuilder().build([(packet, patch), ...])
write_sft_dataset(examples, "data/engineer_sft.jsonl")
```

Teacher filtering keeps only executable, complete, non-negative-reward edits:

```python
from harnyx.training.dataset import filter_training_records
records = [(packet, patch, {"valid": True, "reward": 0.5}), ...]
kept = filter_training_records(records, min_reward=0.0)
```

The `harnyx build-sft-data` CLI performs the same with JSONL inputs.

## Cold-start SFT (paper App. B.1)

| Parameter | Value |
|---|---|
| Base model | Qwen3.5-9B |
| Training examples | 877 |
| Epochs | 2 |
| Context length | 32,768 |
| Global batch size | 24 |
| Optimizer | AdamW |
| Learning rate | 1e-5 |
| LR schedule | cosine |
| Warmup ratio | 0.03 |
| Precision | bf16 |
| Seed | 42 |

```python
from harnyx.training.sft import SFTConfig, train_sft

train_sft(SFTConfig(dataset_path="data/engineer_sft.jsonl", base_model="Qwen3.5-9B"))
```

## Online GRPO (paper App. B.2)

| Parameter | Value |
|---|---|
| Candidates per prompt K | 8 |
| Rollout batch size | 4 prompts |
| Global batch size | 32 sequences |
| Rollout iterations per update | 4 |
| Max policy staleness | 8 |
| Learning rate | 1e-6 (constant) |
| Optimizer | Adam (0.9, 0.98), weight decay 0.1 |
| GRPO clip lower / upper | 0.20 / 0.28 |
| Truncated importance sampling | enabled, max weight 2.0 |
| Entropy / explicit KL | 0 / 0 |
| Rollout temperature / top-p | 0.7 / 0.95 |
| Max prompt / response | 28,672 / 12,288 |
| Seeds (train / rollout) | 1234 / 42 |

Reward = full-batch mean reward change `Δ_B(P)`; invalid, no-op, or incomplete
patches score 0. There is no validity bonus and no learned judge.

```python
from harnyx.training.grpo import GRPOConfig, HarnessPatchReward, train_grpo

reward = HarnessPatchReward(evaluate_patch)   # your sandbox+rerun adapter
train_grpo(GRPOConfig(dataset_path="data/engineer_rl.jsonl"), reward_fn=reward)
```

`evaluate_patch(patch_or_none, metadata) -> float` must validate and rerun the
frozen target on the same task batch.

## Framework note

The reference implementation trains with the authors' **Relax** (GRPO) and
**LLaMA-Factory** (SFT). Harnyx preserves the hyperparameters and the algorithm
interface, and delegates the optimizer step to **TRL** (`SFTTrainer`,
`GRPOTrainer`). Install with `pip install harnyx[train]` to enable these
launchers; otherwise they raise an explicit `ConfigError` rather than pretending
to train.

## Sequence lengths

SFT context 32,768; GRPO prompt 28,672 and response 12,288. The reference
agent-SFT stage uses a 24,576 sequence cutoff and disables target thinking; those
settings belong to target-agent training and are documented here for reference.
