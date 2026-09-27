# Harness-R1 → Harnyx reproduction mapping

This document maps every major component of Harness-R1 to the Harnyx
implementation. It was produced by reading the full paper
(`arXiv:2608.02276v1`, 22 pages including appendices) and the complete official
repository (`DeepExperience/Harness-R1`, ~408 files), not the README alone.

Priority when sources disagree: **explicit paper algorithm → official repository
behaviour → official configuration → tests → engineering decision.** No research
methodology is changed silently. Deviations are called out below.

## Component mapping

| # | Paper concept | Paper section | Official repository | Harnyx implementation | Status |
|---|---|---|---|---|---|
| 1 | Frozen target agent `A` | §3.1 | AgentBench client agents + rollout workers | `harnyx.core.agent.Agent`, `HarnessedAgent` | reproduced (generic) |
| 2 | Base runtime (context, tool mediation, recovery) | §3.1 | `life-harness/AgentBench/src/server/tasks/*/task.py` | `HarnessedAgent` loop + `Harness` protocol | reproduced (decoupled) |
| 3 | Task batch `B` and trajectories `τ` | §3.1 | `runs.jsonl` from rollout workers | `Task`, `Trajectory`, `TrajectoryRecorder` | reproduced (JSONL) |
| 4 | Deterministic failure extraction | §3.1 | `scripts/harness_r1_trace_packet.py` | `TraceFailureAnalyzer`, `compute_signals`, `sanitize` | reproduced |
| 5 | Compact failure packet `s_B` | §3.1, App. A | `build_packet` / `format_trace` | `FailurePacket`, `render_case`, `to_prompt_text` | reproduced |
| 6 | Executable overlay `P` | §3.1, App. C | `harness_r1_patch.py`, `code_runner.py` | `HarnessPatch`, `CodeHook`, `ExecutableHarness` | reproduced |
| 7 | `add_code_hook` only action type | App. A.2, §4.1 | `require_code_hook_only_patch` | `HarnessPatch.from_dict` rejects other types | reproduced |
| 8 | Lifecycle hooks `on_init`, `make_pre_hint`, `on_before_action`, `on_post_step` | §3.1, App. C | four invocation points in each task runtime | `HookNames`; `ExecutableHarness` methods | reproduced |
| 9 | Hook signature `hook(ctx, nb)` | App. A, C | `compile_hook` signature check | `HookPolicy.hook_argument_names` | reproduced |
| 10 | Hook return contracts | App. C | `_normalize_hook_result` | `normalize_effect`, `HookEffect` | reproduced |
| 11 | `block_and_prompt` / `rewrite_action` / `force_action` / `inject_hint` | App. C | `_apply_before_action_effect`, `_apply_post_step_effect` | `HarnessedAgent._resolve_action` + `on_post_step` | reproduced |
| 12 | Per-episode notebook state `nb` | App. C | `code_hook_nb` dict | mutable `nb` mapping passed to every hook | reproduced |
| 13 | AST validation of untrusted code | App. C | `code_runner.compile_hook` (forbidden nodes/names/attrs, string length, ALFWorld leakage) | `harnyx.sandbox.policy.validate_hook_source` | reproduced |
| 14 | Sandboxed execution with budgets | App. C | restricted `__builtins__`, `SIGALRM` timeout, line-count trace | `harnyx.sandbox.runner.LocalSandbox` | reproduced |
| 15 | Runtime failures degrade to no effect | §3.1, App. C | `run_hook` catches and returns `None` | `LocalSandbox.run` returns `None` | reproduced |
| 16 | Same-batch reward `Δ_B(P)` | §3.1 Eq. 1 | `reward_*_patch.py` `delta_average_reward` | `OutcomeReward.score` | reproduced |
| 17 | Invalid / no-op / incomplete ⇒ reward 0 | §3.1, App. B.4 | `_no_patch_reward`, `reject_runtime_noop_patch` | `RewardResult.reason ∈ {invalid_patch, runtime_noop, incomplete_eval}` | reproduced |
| 18 | WebShop shaped reward; ALFWorld/DBBench binary | App. B.4 | per-benchmark reward modules | `OutcomeReward` over arbitrary `EvaluationResult.rewards` | reproduced (generic) |
| 19 | `K = 8` candidates per packet | §3.2, App. B.2 | GRPO `num_generations=8` | `OptimizationConfig.candidates=8` | reproduced (configurable) |
| 20 | Outcome-grounded selection | §3.2 Alg. 1 | reward-gated GRPO | `CandidateSelector` best-reward | reproduced |
| 21 | Group-relative advantage `Â_k` | §3.2 Eq. 3 | Relax GRPO | documented in `harnyx.training.grpo`; TRL adapter | interface reproduced |
| 22 | Clipped surrogate + truncated importance weights | §3.2 Eq. 4 | Relax GRPO; configs | `GRPOConfig` clip 0.20/0.28, TIS weight 2.0 | hyperparameters reproduced |
| 23 | Cold-start SFT objective | §3.2 Eq. 2 | `configs/sft/...`, LLaMA-Factory | `SFTConfig` + `train_sft` (TRL) | interface + data reproduced |
| 24 | Teacher filtering (executable, complete, non-negative reward) | App. B.1 | reward-cache filtering | `filter_training_records` | reproduced |
| 25 | Engineer prompt (system + benchmark context) | App. A.1–A.2 | `schema_prompt`, `harness_r1_edit.py` | `harnyx.engineering.prompt` | reproduced |
| 26 | `prefill_think_patch` response protocol | App. A.3 | `extract_prefilled_think_patch_json_object` | `extract_patch(prefill_think=True)` | reproduced |
| 27 | Data splits | App. D | benchmark-specific manifests | caller-supplied `Task` batches | not bundled |
| 28 | WebShop / ALFWorld / DBBench runtimes | §4.1 | `life-harness` task runtimes | external via `HarnessR1BenchmarkAdapter` | **deviation** (not bundled) |
| 29 | Lifecycle-position ablation | §4.5 | heldout/lifecycle scripts | `harnyx.experiments.ablations` | reproduced (local) |
| 30 | Legacy typed DSL (`set_config`, skills, guard/recovery rules) | App. A.2 (code-hook-only release) | `harness_r1_patch.py` supports 6 action types | **not implemented** | **deviation** (paper's active protocol is code-hook-only) |

## Intentional deviations

1. **Benchmark runtimes are not bundled.** The reference ships WebShop, ALFWorld,
   and AgentBench/DBBench integrations. Harnyx is benchmark-agnostic and provides
   `HarnessR1BenchmarkAdapter` plus a local deterministic evaluator. No benchmark
   assets, models, or datasets are vendored. This is why paper-level numbers
   cannot be reproduced in this repository (see `docs/research.md`).
2. **Execution backend.** The reference executes AST-validated hooks in-process
   under a `SIGALRM` timeout. Harnyx reproduces that exactly as `LocalSandbox`
   (default for rollout speed), and **additionally** offers `SubprocessSandbox`
   (fresh interpreter, sanitized env, `RLIMIT_AS`/`RLIMIT_CPU`, wall timeout)
   for pre-flight validation and stronger isolation. The added boundary does not
   change hook semantics.
3. **Legacy action types.** The repository contains six patch action types for
   older experiments. The paper's active protocol and the released engineer
   prompt are `add_code_hook` only, enforced by `require_code_hook_only_patch`.
   Harnyx implements only the released protocol.
4. **Training framework.** The reference trains with the authors' Relax for GRPO
   and LLaMA-Factory for SFT. Harnyx exposes the identical hyperparameters
   (`SFTConfig`, `GRPOConfig`) and delegates updates to TRL when installed. The
   algorithmic interface (same-batch outcome reward, K candidates, group-relative
   advantages, clipped objective) is preserved; the framework is not vendored.
5. **Harnyx extensions.** Patch caching, failure clustering, and an explicit
   regression suite are Harnyx additions beyond the paper. They are opt-in
   (`OptimizationConfig.patch_cache`, `cluster_failures`, `reject_regressions`)
   and are separated from the reproduction path.

## Reward definition (verified Eq. 1)

```
Δ_B(P) = (1/n) Σ_i ( R_i^P − R_i^0 )
r(B,P) = Δ_B(P)   if the patch is valid and evaluation is complete
         0        otherwise
```

`OutcomeReward` implements this exactly, including negative rewards for
regressive patches and zero for invalid/no-op/incomplete evaluation. There is no
LLM judge anywhere in the reward path.

## Hook semantics (verified App. C / `code_runner.py`)

| Hook | Context available | Permitted return |
|---|---|---|
| `on_init` | observation, step, task, state, admissible | `{"skills": [...], "tool_hint": ...}` |
| `make_pre_hint` | + page/state evidence | `{"message": ...}` |
| `on_before_action` | + proposed `action` | `block_and_prompt`, `rewrite_action`, `force_action` |
| `on_post_step` | + executed `action`, returned observation | `inject_hint`, `force_action` |

DBBench accepts soft guidance and blocking but does not execute SQL rewrites or
forced commits (paper App. C); Harnyx leaves that restriction to the benchmark
adapter/context.
