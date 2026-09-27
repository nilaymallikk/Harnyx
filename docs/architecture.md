# Architecture

Harnyx is built in layers with one rule: no layer imports upward, and no layer
depends on a specific model provider, benchmark, or training framework.

```text
                 ┌──────────────────────────────────────────────┐
                 │                    cli/                       │
                 └───────────────┬──────────────────────────────┘
        ┌────────────────────────┼───────────────────────────────┐
        ▼                        ▼                                ▼
┌───────────────┐      ┌──────────────────┐            ┌──────────────────┐
│ optimization/ │◄────►│   engineering/   │            │    adapters/     │
│ optimizer     │      │ patch, validate, │            │ nyvero           │
│ failure pkt   │      │ engineer, prompt │            └──────────────────┘
│ reward, select│      └────────┬─────────┘
└──────┬────────┘               │
       │                        ▼
       │                ┌───────────────┐
       │                │   sandbox/    │  AST policy, local + subprocess
       │                └───────┬───────┘
       ▼                        ▼
┌──────────────────────────────────────────────────────────────────────┐
│                               core/                                    │
│  Task  Trajectory  Agent/HarnessedAgent  Harness hook contract  Result │
└──────────────────────────────────────────────────────────────────────┘
        ▲                        ▲                                ▲
        │                        │                                │
┌───────┴──────┐        ┌────────┴────────┐            ┌──────────┴───────┐
│ evaluation/  │        │    adapters/    │            │      llm/        │
│ Local, H-R1  │        │     nyvero      │            │ provider, openai │
└──────────────┘        └─────────────────┘            └──────────────────┘
```

## Layer responsibilities

- **core** — data and protocols only. `Task`, `Trajectory`,
  `TrajectoryRecorder`, `HookContext`, `HookEffect`, `Harness`,
  `ExecutableHarness`, `HarnessedAgent`, `AgentResult`, `EvaluationResult`.
  Zero dependencies beyond the standard library.
- **sandbox** — the static AST policy (`policy.py`), the compiler and in-process
  runner (`runner.py`), the process-isolated runner (`isolation.py`), and
  budgets (`limits.py`).
- **engineering** — the patch representation and parser (`patch.py`), static
  validation (`validation.py`), and engineers (`harness_engineer.py`) and the
  prompt protocol (`prompt.py`).
- **optimization** — `FailurePacket` construction (`failure_analysis.py`),
  candidate generation (`patch_generation.py`), the outcome reward (`reward.py`),
  regression-aware selection (`selection.py`), and the loop (`optimizer.py`).
- **evaluation** — `LocalEvaluator`, `HarnessR1BenchmarkAdapter`, and
  `RunDirectory` for reproducible artifacts.
- **adapters** — translate external agents/runtimes into Harnyx interfaces
  (`nyvero.py`).
- **llm** — provider protocol plus an OpenAI-compatible stdlib client.
- **cli** — argparse front end; every command does real work.

Research-only code (SFT/GRPO training, ablation arms, the random baseline
engineer) lives in `research/` at the repository root. It depends on the
installed package but is **not** part of the shipped wheel or sdist.

## The frozen/editable boundary

```text
frozen policy  ── proposes ──►  action
                                  │
editable harness ── on_before_action ──►  block | rewrite | force | pass
                                  │
environment  ── executes ──►  observation
                                  │
editable harness ── on_post_step ──►  inject_hint | force_action | pass
```

The harness observes and returns structured effects. The host runtime alone
executes environment actions, which is why a generated patch cannot seize
arbitrary host control.

## Determinism

- `harnyx.core.types.canonical_json` sorts keys and uses compact separators.
- Run directories, failure packets, candidates, rewards, and reports are JSON/JSONL.
- The reward is deterministic whenever the evaluator is; the toy demo and all
  tests are fully deterministic.
