# NOVA

**Learn to improve executable AI-agent harnesses from failure trajectories.**

NOVA is a clean, agent-agnostic Python library implementing the
[Harness-R1](https://arxiv.org/abs/2608.02276) methodology: mine a batch of
target-agent failures, have a harness engineer propose an executable runtime
patch, sandbox it, rerun the *same* tasks, and accept it only when a real
outcome improvement is measured.

```text
target agent rollout
  -> batch failure packet
  -> harness engineer  (LLM or scripted)
  -> parse <think>...</think><patch>...</patch>
  -> AST validate + sandbox
  -> rerun the frozen target on the same task identities
  -> reward = patched performance - baseline performance
  -> accept / reject (with regression protection) -> versioned harness
```

NOVA is not an agent framework. It is the *optimization loop around* an agent:
the harness is the editable object, not the model weights.

---

## Relationship to Harness-R1

NOVA is an independent implementation of the method in
**"Harness-R1: Learning to Edit Executable Runtime Harnesses from Agent Failure
Trajectories"** (Shao et al., 2026). The paper/repository and NOVA map
component-by-component in [`docs/reproduction.md`](docs/reproduction.md).

- **Reproduced faithfully:** the four executable lifecycle hooks
  (`on_init`, `make_pre_hint`, `on_before_action`, `on_post_step`), the
  `hook(ctx, nb)` contract, the `add_code_hook`-only patch protocol, the
  same-batch outcome reward (Eq. 1), the `K = 8` candidate group, AST/sandbox
  restrictions, the engineer prompt/response protocol, and the
  SFT + GRPO hyperparameters.
- **Deliberate deviations:** benchmark runtimes are not bundled (use an adapter);
  training delegates to TRL instead of vendoring Relax; only the released
  code-hook protocol is implemented (not the legacy six-action DSL). See
  [`docs/reproduction.md`](docs/reproduction.md#intentional-deviations).
- **NOVA extensions (opt-in):** patch caching, failure clustering, and an
  explicit regression suite. See [`docs/research.md`](docs/research.md).

Reference implementation: <https://github.com/DeepExperience/Harness-R1>
(used for behavioural verification only; no source is copied).

## Architecture

```text
nova/
├── core/          Task, Trajectory, Agent, Harness (hook contract), Result
├── engineering/   HarnessPatch, parser, PatchValidator, HarnessEngineer, prompts
├── sandbox/       AST policy, LocalSandbox, SubprocessSandbox, limits
├── optimization/  FailurePacket, PatchGenerator, OutcomeReward, selection, optimizer
├── evaluation/    LocalEvaluator, HarnessR1BenchmarkAdapter, metrics, run reports
├── adapters/      Generic adapters + Nyvero adapter (no Nyvero dependency)
├── llm/           Provider protocol, OpenAI-compatible client, scripted provider
├── training/      SFT dataset/config, GRPO config + TRL bridge
├── experiments/   A–F ablation harness
├── demo/          Deterministic toy end-to-end
└── cli/           nova <command>
```

The hard boundary: **frozen policy** (never edited) vs **editable harness**
(only four hooks, only structured effects). A hook never executes an environment
action itself; the host runtime interprets its return value.

## Installation

```bash
pip install -e .            # core, no dependencies
pip install -e ".[yaml]"    # + YAML configs
pip install -e ".[train]"   # + trl/transformers/datasets for SFT & GRPO
pip install -e ".[dev]"     # + pytest/ruff/mypy
```

Requires Python ≥ 3.11. Core has **zero runtime dependencies**.

## Minimal example

The deterministic toy demo needs no model, GPU, or benchmark assets:

```bash
nova run
# baseline success 0/1, patched success 1/1, engineer reward +1.000
```

Or in Python:

```python
from nova.demo.toy import run_demo

print(run_demo("runs", candidates=3))

from nova import (
    ExecutableHarness, FailurePacket, LocalEvaluator, LocalSandbox,
    ScriptedHarnessEngineer, HarnessOptimizer,
)
```

A full optimization over your own agent:

```python
from nova import HarnessOptimizer, LocalEvaluator, LLMHarnessEngineer
from nova.llm.openai import OpenAICompatibleProvider
from nova.optimization.optimizer import OptimizationConfig

provider = OpenAICompatibleProvider(
    base_url="http://localhost:8000/v1",  # OpenAI / OpenRouter / vLLM / SGLang
    model="Qwen3.5-9B-engineer",
    env_key="NOVA_ENGINEER_API_KEY",
)
engineer = LLMHarnessEngineer(provider, benchmark="mybench")

optimizer = HarnessOptimizer(
    agent,                       # any object with .run(task, harness, recorder)
    engineer,
    evaluator=LocalEvaluator(benchmark="mybench"),
    benchmark="mybench",
    config=OptimizationConfig(candidates=8, iterations=3),
    run_dir="runs",
)
result = optimizer.optimize(tasks)          # tasks: list[nova.Task]
print(result.final.mean_reward - result.baseline.mean_reward)
```

See [`docs/quickstart.md`](docs/quickstart.md) and
[`docs/custom-agent.md`](docs/custom-agent.md).

## Nyvero example

Nyvero is never a dependency of NOVA core. The adapter targets a documented
duck-typed contract (see [`docs/nyvero.md`](docs/nyvero.md)):

```python
from nova.adapters.nyvero import NyveroAgentAdapter, NyveroHarnessAdapter

nova_agent = NyveroAgentAdapter(nyvero_agent, benchmark="nyvero")
nova_harness = NyveroHarnessAdapter(nyvero_harness)  # expose Nyvero's harness
result = nova_agent.run(task, harness=nova_harness)
```

## Reproduction instructions

```bash
# Deterministic local end-to-end (works in CI, no model)
nova run

# Full local reproduction: toy loop + A-F ablations + security smoke
python examples/reproduction/run_local_reproduction.py --run-root runs/reproduction

# Full paper reproduction (WebShop/ALFWorld/DBBench) is documented, not runnable
# here because benchmark assets, Qwen3.5 models, and 8xH800 are unavailable.
```

The honest reproduction status — expected paper numbers vs what was actually
observed on the available hardware — is in [`docs/research.md`](docs/research.md).
No benchmark result in this repository is fabricated.

To reproduce the paper protocol on real benchmark runtimes, point
`HarnessR1BenchmarkAdapter` at the reference AgentBench runtimes (or any
harness-aware runner) and use the released reference task splits.

## Security model

Generated harness code is untrusted. Every candidate passes:

```text
parse -> schema validation -> AST policy -> sandbox smoke test -> execution
```

`nova.sandbox.policy` rejects imports, filesystem/network/subprocess access,
dynamic evaluation (`eval`/`exec`/`compile`), dunder/attribute escapes,
`getattr`/`setattr`/`globals`/`locals`, async/class/with/lambda/while/yield,
generators, raise, and benchmark-answer leakage (e.g. numbered ALFWorld
instances). Execution uses a restricted builtin set, a wall-clock timeout, and a
line budget; runtime failures degrade to *no intervention*. `SubprocessSandbox`
adds process isolation, a sanitized environment (no credentials), and
`RLIMIT_AS`/`RLIMIT_CPU`. See [`docs/sandbox.md`](docs/sandbox.md).

No API key is ever hard-coded or logged; providers read them from environment
variables.

## Benchmarks and adapters

NOVA ships a deterministic local evaluator and a benchmark adapter. The reference
paper's WebShop (500 tasks), ALFWorld (500 tasks), and DBBench (300 tasks) are
supported through `HarnessR1BenchmarkAdapter` but their runtimes are not bundled.
It is deliberately *not* a core dependency of NOVA.

## Limitations

- **No paper-level benchmark reproduction here.** The benchmark runtimes, model
  weights, and 8×H800 hardware are not available in this environment. See
  `docs/research.md` for the exact gap and likely causes.
- Training stages expose the reference hyperparameters and a TRL launcher; the
  authors' Relax/LLaMA-Factory stack is not vendored.
- The legacy six-action DSL from the repository is not implemented; only the
  released `add_code_hook` protocol is.
- Reward is transductive (same tasks before/after), exactly as in the paper.
- The in-process sandbox is hardened but not a full OS sandbox; use
  `SubprocessSandbox` when stronger isolation is required.

## Citation

```bibtex
@misc{shao2026harnessr1learningeditexecutable,
      title={Harness-R1: Learning to Edit Executable Runtime Harnesses from Agent Failure Trajectories},
      author={Shuai Shao and Kangning Zhang and Qingyao Li and Shijian Wang and Hao Wang and Wenxiang Jiao and Yuan Lu and Yi Guo and Weiwen Liu and Weinan Zhang},
      year={2026},
      eprint={2608.02276},
      archivePrefix={arXiv},
      primaryClass={cs.AI},
      url={https://arxiv.org/abs/2608.02276},
}
```

## License

Apache-2.0. NOVA is an independent implementation; see [`NOTICE`](NOTICE) for
attribution of the Harness-R1 reference implementation, Life-Harness/AgentBench,
Relax, and the benchmark environments.
