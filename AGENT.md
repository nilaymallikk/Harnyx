# AGENT.md — working guide for coding agents in Harnyx

This file tells an autonomous coding agent (or a new contributor) how this
repository is structured, what must never break, and how to make changes.

## 1. What this project is

**Harnyx** is a Python library implementing the **Harness-R1** methodology
("Learning to Edit Executable Runtime Harnesses from Agent Failure Trajectories",
arXiv:2608.02276). It learns to improve the *runtime harness* around a frozen
agent from failure trajectories — not the model weights.

Pipeline: failure packet → harness engineer → executable patch → AST/sandbox
validation → rerun the *same* tasks → same-batch outcome reward → accept/reject →
versioned harness.

It is **not** an agent framework. Do not add task-solving logic to the harness.

## 2. Non-negotiable invariants

Break any of these and the change is wrong, regardless of tests passing.

1. **Frozen policy / editable harness boundary.** The agent policy is never
   edited. The only editable surface is four lifecycle hooks.
2. **Hooks return structured effects; the host runtime acts.** A hook never
   executes an environment action itself. Effects are exactly:
   `block_and_prompt`, `rewrite_action`, `force_action` (pre-action) and
   `inject_hint`, `force_action` (post-feedback), plus `skills`/`tool_hint`
   (`on_init`) and `message` (`make_pre_hint`).
3. **Reward is outcome-grounded, never a judge.**
   `Δ = mean(patched reward) − mean(baseline reward)` over the same task ids.
   Invalid, no-op, or incomplete evaluations score `0`. A negative delta is a
   real result and must be preserved.
4. **Untrusted code is sandboxed.** Never execute generated patch code without
   passing the AST policy and sandbox. Never weaken
   `src/harnyx/sandbox/policy.py` on the reproduction path.
5. **No fake functionality.** Every public function does real work or raises an
   explicit error naming the missing capability. Never stub with
   `print("done")` or `return "optimized"`.
6. **No fabricated research results.** Do not invent benchmark numbers. The
   honest status lives in `docs/research.md`.
7. **No secrets.** API keys come from environment variables only. Never
   hard-code, log, or commit a key.

## 3. Ground truth: paper and reference implementation

The behavior is verified against two sources, in this priority order:

1. the paper's explicit algorithm,
2. the official repository behavior,
3. the official configuration,
4. the reference tests,
5. a reasonable engineering decision.

The component-by-component mapping and every intentional deviation are in
[`docs/reproduction.md`](docs/reproduction.md). **If you change behavior that
touches the method, update that file.** The reference repository is
`DeepExperience/Harness-R1` (URL recorded in the top-level
`https github com DeepExperience.txt`).

Do not silently invent methodology. If the paper and repo disagree, document it
and follow the released implementation unless the paper explicitly overrides.

## 4. Environment and commands

Requires **Python ≥ 3.11**. Core has **zero runtime dependencies**.

```bash
# Install for development
pip install -e ".[dev]"        # adds pytest, ruff, mypy, PyYAML
pip install -e ".[yaml]"       # YAML configs only

# Test / lint / type-check (run before every commit)
pytest -q                      # 103 tests, all deterministic, no network/GPU
pytest research -q             # research/training + ablation tests
ruff check src tests research examples
mypy                           # config in pyproject.toml

# Run the deterministic end-to-end demo (no model, no GPU)
harnyx run

# Full local reproduction: toy loop + A-F ablations + security smoke
python examples/reproduction/run_local_reproduction.py --run-root runs/reproduction
```

Test and lint must both be green before committing. The test suite must stay
offline and fast.

## 5. Repository map

```text
src/harnyx/
├── core/          Task, Trajectory, Agent/HarnessedAgent, Harness hook contract, Result
├── sandbox/       policy.py (AST), runner.py (LocalSandbox), isolation.py (SubprocessSandbox), limits.py
├── engineering/   patch.py, validation.py, harness_engineer.py, prompt.py
├── optimization/  failure_analysis.py, patch_generation.py, reward.py, selection.py, optimizer.py
├── evaluation/    evaluator.py, harness_r1.py (benchmark adapter), reports.py
├── adapters/      nyvero.py
├── llm/           provider.py, openai.py, local.py
├── demo/          toy.py (deterministic end-to-end)
└── cli/           main.py
tests/             13 files, mirror the shipped modules
docs/              required reading: reproduction.md, research.md, sandbox.md, architecture.md
examples/          plugins/ (CLI plugin), patches/, reproduction/
research/          NOT shipped: training/, experiments/, random_engineer.py, tests/
configs/           harnyx.example.yaml
```

Layering rule: `core` imports nothing from the layers above. `sandbox` imports
`core`. `engineering` imports `sandbox` + `core`. `optimization` imports
`engineering` + `core` + `evaluation`. Never import upward.

## 6. How to make common changes

**Add a hook effect or refine normalization** → `core/harness.py`
(`HookEffect`, `ALLOWED_EFFECT_KINDS`) and `sandbox/runner.py`
(`normalize_effect`). Add a test in `tests/test_sandbox.py`. Keep the
degrade-to-`None` behavior for malformed returns.

**Change the sandbox/AST policy** → `sandbox/policy.py`. Any loosening is a
security change: add an adversarial test to `tests/test_sandbox.py` and
document the rationale in `docs/sandbox.md`. Never allow imports, I/O,
subprocess, network, `eval`/`exec`/`compile`, `getattr`/`setattr`, or dunder
access.

**Add a harness engineer** → implement `generate_patch` /
`generate_candidates` (see `engineering/harness_engineer.py`). Sample each
candidate independently; do not cache model state across candidates. Register it
in `engineering/__init__.py`.

**Add an evaluator or benchmark adapter** → implement the `Evaluator` protocol.
Rewards must be keyed by `Task.id` so baseline and patched runs pair per task.
Incomplete runs must surface in `EvaluationResult.errors`. See
`evaluation/harness_r1.py` for external benchmarks — never bundle benchmark
runtimes into core.

**Add an optimizer feature** → `optimization/optimizer.py`. Separate
reproduction behavior from Harnyx extensions behind config flags, and label
extensions as such in `docs/research.md`.

**Add a CLI command** → `cli/main.py`. It must actually work and return a
meaningful exit code; add a test in `tests/test_cli.py`. Do not add commands
that only print.

**Add a training feature** → `research/training/`. This code is not shipped in
the package; lazy-import heavy frameworks and raise `ConfigError` with an
actionable message when the extra is missing.

## 7. Testing requirements

- Non-trivial logic (a branch, loop, parser, money/security path) leaves **one
  runnable check** behind. Trivial one-liners do not need a test.
- Deterministic only: no network, no GPU, no wall-clock dependence, no
  randomness without a fixed seed.
- Security changes require adversarial cases: forbidden import, filesystem,
  subprocess, network, dynamic execution, timeout, runtime exception, malformed
  model output, empty patch, duplicate patch.
- Prefer synthetic `EvaluationResult`s for unit tests over full runs; use
  `tests/test_optimizer.py` and `examples/reproduction/` for end-to-end.

## 8. Code conventions

- Python 3.11+, `from __future__ import annotations`, full type hints.
- Docstrings on public APIs; `errors.py` for the typed error hierarchy; no
  silent exception swallowing (runtime failures that must degrade to no-op are
  explicit and documented).
- Deterministic serialization via `core/types.py` (`canonical_json`, `write_json`,
  `write_jsonl`). Never emit unordered or timestamp-dependent JSON in tests.
- Keep files and functions small; dependency injection over hidden global state.
- No new runtime dependencies in `core`. Justify any dependency added anywhere
  else in the PR description.
- Use `logging` via module loggers, not `print`, in library code (CLI may print).

## 9. Git and CI

- Work on `main`; the remote is `origin` → `nilaymallikk/Harnyx`.
- Never force-push over existing history; the repo keeps a linear history.
- Commit style: imperative subject, body explaining *why*, and a note if the
  reproduction mapping changed.
- CI (`.github/workflows/ci.yml`) runs `ruff check src tests`, `pytest -q`, and
  the toy demo on Python 3.11 and 3.12. Keep it green.
- `.pre-commit-config.yaml` runs ruff + pytest.

## 10. Gotchas

- The `HarnessedAgent` loop appends assistant actions and observations to
  `messages`; a policy that needs history reads from there (see
  `demo/toy.py`).
- `harness-v0` is the implicit base runtime; the first recorded patch is
  `harness-v1`.
- `OutcomeReward(require_complete=True)` scores `0` when any task errored — make
  sure the evaluator populates `errors`.
- `LocalSandbox` runs in-process under a `SIGALRM` timeout; use
  `SubprocessSandbox` when a real process boundary is needed.
- The distribution is `harnyx`, the import is `harnyx`, the CLI is `harnyx`
  (post-rename; no `nova` references remain anywhere).
- Some committed files (`examples/`, `configs/`) are user-facing; changing them
  requires keeping `docs/quickstart.md` accurate.
