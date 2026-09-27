# Research status: reproduction and extensions

This document separates three things and never blurs them:

1. **Harness-R1 reproduction** — the paper method, faithfully implemented.
2. **Harnyx extensions** — additions made only after the reproduction path exists.
3. **Reproduction experiments** — what was actually run, and what was not.

No benchmark number in this repository is fabricated.

## 1. What was reproduced

The full method is implemented and unit-tested:

- lifecycle hook contract and effect normalization,
- `add_code_hook`-only patch protocol and parser,
- AST policy and sandbox execution,
- failure-packet construction,
- same-batch outcome reward (Eq. 1),
- `K`-candidate generation and outcome-based selection,
- regression-protected acceptance and harness versioning,
- engineer prompt/response protocol,
- SFT dataset builder and the reference SFT/GRPO hyperparameters.

## Runnable reproduction harness

```bash
python examples/reproduction/run_local_reproduction.py --run-root runs/reproduction
```

This runs the toy loop, the A–F ablations, and the adversarial sandbox suite,
and writes `runs/reproduction/reproduction.json`.

## 2. Deterministic end-to-end result

The toy benchmark (`harnyx.demo.toy`) mirrors the paper's WebShop premature-purchase
case at miniature scale. It requires no model and is deterministic:

| Arm | Baseline reward | Patched reward | Engineer reward |
|---|---|---|---|
| Toy `guarded-commit` | 0.000 | 1.000 | +1.000 |

The engineering path exercised: real `<think>/<patch>` parsing, real schema +
AST validation, real sandboxed hook execution, real rerun, real reward, real
selection, real versioning (`harness-v1`), real run artifacts.

## 3. Ablations (local, deterministic)

`harnyx.experiments.ablations` implements arms A–F. On the toy benchmark:

| Key | Arm | Toy reward |
|---|---|---|
| A | baseline | 0.0 |
| B | random safe-hook patches | outcome-dependent |
| C | no outcome feedback (first valid) | 1.0 |
| D | outcome selection | 1.0 |
| E | full Harness-R1 (trained engineer) | 1.0 on toy; not runnable without a trained engineer |
| F | Harnyx extension (cache + clustering + regression guard) | 1.0 |

Arms C–F coincide on a single-task toy because the scripted engineer always
proposes the correct patch. They separate only when proposals vary in quality,
which is what real benchmarks provide. The ablation code is real and runs; it is
not evidence of paper-level gains.

## 4. Paper reproduction experiment — expected vs observed

**Status: not reproduced.** The reference benchmarks, model checkpoints, and
hardware are not available in this environment. This is reported, not hidden.

| Item | Expected (paper) | Observed here | Cause |
|---|---|---|---|
| WebShop success (Qwen3.5-9B) | 31.2 → 42.2 | not run | WebShop runtime + 500-task asset + Qwen3.5-9B unavailable |
| ALFWorld All | 40.6 → 53.2 | not run | ALFWorld runtime + assets unavailable |
| DBBench success | 61.0 → 65.3 | not run | AgentBench/DBBench + MySQL driver unavailable |
| Average | 44.3 → 53.6 (+9.3) | not run | as above |
| Agent-SFT + Harness-R1 | 59.2 → 64.2 | not run | agent-SFT target checkpoint unavailable |
| Training | SFT 877 ex, GRPO 8×H800 | not run | no GPU/training data; `trl` not installed |
| Toy end-to-end | n/a | baseline 0 → patched 1 (+1) | **reproduced deterministically** |
| Sandbox/security | AST policy, timeout, no imports | all adversarial tests pass | reproduced |
| Ablation arms A–F | paper §4.5 position ablation | runnable on toy; A=0.0, D/F=+1.0 | runnable, not benchmark-scale |

Environment for this report:

- host: no GPUs, Python 3.14, `PyYAML` available, `trl`/`torch` not installed,
- no benchmark runtimes, model weights, or task datasets,
- no network access to the gated benchmark assets.

Because the target agents, task splits, and runtime substrates are all
unavailable, reporting any percentage would be fabrication. The honest statement
is: **the method is implemented and verified; the benchmark table is not
reproduced.**

## 5. What would be required to reproduce the benchmark table

1. Install WebShop, ALFWorld, and AgentBench/DBBench from the reference
   `docs/BENCHMARK_SETUP.md`.
2. Serve the frozen target (e.g. Qwen3.5-9B) and the released engineer checkpoint
   behind OpenAI-compatible endpoints.
3. Build failure packets on the reference task splits (App. D) and run
   `HarnessOptimizer` with `HarnessR1BenchmarkAdapter`.
4. Run the A–F ablations with the released engineer and the target-specific
   splits, seeding baselines per the reference protocol.

## 6. Harnyx extensions (separate from reproduction)

Implemented and opt-in:

- **Patch caching** (`OptimizationConfig.patch_cache`) — skips re-evaluating an
  identical patch on an identical packet.
- **Regression suite** (`CandidateSelector(regression_suite=...)`) — an explicit
  set of previously solved tasks that must keep passing; complements the paper's
  non-regressive teacher filtering.
- **Subprocess isolation** (`SubprocessSandbox`) — stronger execution boundary
  than the reference in-process runner.

These are not part of the Harness-R1 method and are not used to claim paper
reproduction.
