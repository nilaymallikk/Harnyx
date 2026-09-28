<div align="center">
  <img src="https://raw.githubusercontent.com/nilaymallikk/Harnyx/main/assets/logo.png" alt="Harnyx" width="460">
</div>

**Learn to improve executable AI-agent harnesses from failure trajectories.**

Harnyx is an agent-agnostic Python library implementing the method from the paper
**Harness-R1: Learning to Edit Executable Runtime Harnesses from Agent Failure
Trajectories** ([Shao et al., 2026, arXiv:2608.02276](https://arxiv.org/abs/2608.02276)).
It mines a batch of target-agent failures, has an engineer model propose an
executable runtime patch, sandboxes it, reruns the *same* tasks, and accepts it
only when a real outcome improvement is measured.

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

Harnyx is not an agent framework. It is the *optimization loop around* an agent:
the harness is the editable object, not the model weights.

## Why you'd use it

Your agent's model is frozen — an API, or a self-hosted checkpoint you are not
going to fine-tune — but it still fails in recurring, systematic ways: wrong
tool arguments, dropped state, protocol violations, repeated actions, no
recovery after an error. Today you patch that by hand (prompts, guards, retry
logic) with no evidence it actually helps.

Harnyx automates that loop:

- **It edits the runtime, not the model.** The editable surface is four
  lifecycle hooks around your frozen policy: initialize context, hint before a
  decision, validate/rewrite/veto an action before it runs, recover after
  feedback.
- **An engineer model writes the edits** from your own failure traces, so the
  fix targets the failures you actually have instead of a generic prompt.
- **Every edit is measured, not judged.** Harnyx reruns the *same* tasks with and
  without the patch and keeps it only if task success/reward improves.
- **It refuses to make things worse.** A patch that breaks a previously solved
  task is rejected (regression protection); every accepted patch is versioned
  and reversible.
- **You get a tiny artifact to load at runtime.** The output is a validated
  code-hook patch (`harness-vN`) you ship alongside your agent.

Use it when you can measure task outcomes and want your agent's success rate to
improve automatically, without training the model.

![Harnyx architecture](https://raw.githubusercontent.com/nilaymallikk/Harnyx/main/assets/architecture.png)

## What it actually does (a concrete example)

Say your support agent has tools `lookup_order`, `issue_refund`, and `reply`,
and its failure traces show a pattern: it calls `issue_refund` before a
successful `lookup_order`, so the refund errors and the agent loops.

The engineer reads those traces and writes **one hook**:

```python
def hook(ctx, nb):                          # on_before_action
    action = str((ctx.get("action") or {}).get("name") or "")
    verified = (ctx.get("state") or {}).get("order_verified")
    if action == "issue_refund" and not verified:
        return {"kind": "block_and_prompt",
                "message": "Look up the order before refunding."}
    return None
```

Harnyx then reruns the **same tasks** with and without that hook. If success
improves and nothing that used to pass breaks, it is saved as `harness-v1` and
becomes a tiny artifact you load next to your agent:

```python
harness = ExecutableHarness.from_patch(patch, sandbox=LocalSandbox())
agent.run(task, harness=harness)
```

No model change, no new prompt framework - one reviewed, reversible code hook.

---

## Demo

![Harnyx CLI demo](https://raw.githubusercontent.com/nilaymallikk/Harnyx/main/assets/demo.gif)

Recorded on this repository's deterministic toy benchmark: baseline reward `0.0`
-> patched reward `1.0`, accepting `harness-v1` (no model, GPU, or benchmark
assets required). Video: [`assets/demo.mp4`](https://github.com/nilaymallikk/Harnyx/blob/main/assets/demo.mp4).

---

## Relationship to Harness-R1

Harnyx is an independent implementation of the method in
**"Harness-R1: Learning to Edit Executable Runtime Harnesses from Agent Failure
Trajectories"** (Shao et al., 2026). The paper/repository and Harnyx map
component-by-component in [`docs/reproduction.md`](https://github.com/nilaymallikk/Harnyx/blob/main/docs/reproduction.md).

- **Reproduced faithfully:** the four executable lifecycle hooks
  (`on_init`, `make_pre_hint`, `on_before_action`, `on_post_step`), the
  `hook(ctx, nb)` contract, the `add_code_hook`-only patch protocol, the
  same-batch outcome reward (Eq. 1), the `K = 8` candidate group, AST/sandbox
  restrictions, the engineer prompt/response protocol, and the
  SFT + GRPO hyperparameters.
- **Deliberate deviations:** benchmark runtimes are not bundled (use an adapter);
  training delegates to TRL instead of vendoring Relax; only the released
  code-hook protocol is implemented (not the legacy six-action DSL). See
  [`docs/reproduction.md`](https://github.com/nilaymallikk/Harnyx/blob/main/docs/reproduction.md#intentional-deviations).
- **Harnyx extensions (opt-in):** patch caching, failure clustering, and an
  explicit regression suite. See [`docs/research.md`](https://github.com/nilaymallikk/Harnyx/blob/main/docs/research.md).

Reference implementation: <https://github.com/DeepExperience/Harness-R1>
(used for behavioural verification only; no source is copied).

## Architecture

```text
harnyx/
├── core/          Task, Trajectory, Agent, Harness (hook contract), Result
├── engineering/   HarnessPatch, parser, PatchValidator, HarnessEngineer, prompts
├── sandbox/       AST policy, LocalSandbox, SubprocessSandbox, limits
├── optimization/  FailurePacket, PatchGenerator, OutcomeReward, selection, optimizer
├── evaluation/    LocalEvaluator, HarnessR1BenchmarkAdapter, run reports
├── adapters/      Nyvero adapter (no Nyvero dependency)
├── llm/           Provider protocol, OpenAI-compatible client, scripted provider
├── demo/          Deterministic toy end-to-end
└── cli/           harnyx <command>
```

Research-only code (SFT/GRPO training, ablations, the random baseline engineer)
lives in `research/` at the repository root and is **not** part of the
installed package.

The hard boundary: **frozen policy** (never edited) vs **editable harness**
(only four hooks, only structured effects). A hook never executes an environment
action itself; the host runtime interprets its return value.

## Install

```bash
pip install harnyx        # or: uv add harnyx
```

Requires Python ≥ 3.11. The core library has **zero runtime dependencies**.
Optional extra: `harnyx[yaml]` for YAML config files.

## Verify the install (self-test, no model needed)

This is a deterministic offline self-test that proves the whole pipeline works
(parse → validate → sandbox → rerun → reward → version). It is **not** how you
use the library in your app.

```bash
harnyx run
# baseline success 0/1  ->  patched success 1/1, engineer reward +1.000, harness-v1
```

## Use it in your agent

You provide three things: your **environment**, your **policy**, and a **task
batch**. Harnyx supplies the harness, the engineer loop, the sandbox, the
outcome reward, and selection.

```python
from harnyx import Action, StepResult, Task, HarnessedAgent
from harnyx import HarnessOptimizer, LocalEvaluator, LLMHarnessEngineer
from harnyx.llm.openai import OpenAICompatibleProvider
from harnyx.optimization.optimizer import OptimizationConfig

class MyEnv:
    """Runs one task and reports an objective outcome."""
    name = "support"
    def reset(self, task: Task) -> str: ...          # -> initial observation
    def step(self, action: Action) -> StepResult: ... # -> next observation (+done)
    def state(self) -> dict: ...                      # exposed to hooks as ctx["state"]
    def predicates(self) -> dict: ...                 # exposed as ctx["predicates"]
    def success(self) -> bool: ...                    # your success criterion
    def episode_reward(self) -> float: ...            # 0/1, or a shaped score
    def admissible_actions(self) -> list[str]: ...

class MyPolicy:
    """Your frozen LLM. It only decides; it never edits the harness."""
    name = "my-policy"
    def act(self, messages, *, step, admissible) -> Action:
        # call your model here and return Action(name=..., arguments=...)
        ...

agent = HarnessedAgent(MyPolicy(), MyEnv(), benchmark="support", max_steps=12)

# The engineer is an LLM that reads failure traces and writes harness patches.
engineer = LLMHarnessEngineer(
    OpenAICompatibleProvider(
        base_url="http://localhost:8000/v1",   # OpenAI / OpenRouter / vLLM / SGLang
        model="your-engineer-model",
        env_key="HARNYX_ENGINEER_API_KEY",
    ),
    benchmark="support",
)

# Mine failures -> propose patches -> rerun the same tasks -> keep what improves.
result = HarnessOptimizer(
    agent,
    engineer,
    evaluator=LocalEvaluator(benchmark="support"),
    benchmark="support",
    config=OptimizationConfig(candidates=8, iterations=3),
    run_dir="runs",
).optimize(tasks)          # tasks: list[Task] with stable .id values

print(result.baseline.mean_reward, "->", result.final.mean_reward)
```

Already have an agent loop? Wrap it instead of using `HarnessedAgent`: call
`harness.on_init`, `make_pre_hint`, `on_before_action`, and `on_post_step` at
your lifecycle points. See
[`docs/custom-agent.md`](https://github.com/nilaymallikk/Harnyx/blob/main/docs/custom-agent.md).

## Ship an accepted patch

The optimizer writes `runs/<timestamp>/accepted_patch.json` and a versioned
harness. Load the patch into your production agent — no model change, and the
patch is inert until you choose to install it:

```python
import json
from harnyx import ExecutableHarness, LocalSandbox, HarnessPatch

raw = json.load(open("runs/<timestamp>/accepted_patch.json"))["patch"]
patch = HarnessPatch.from_dict(raw)
harness = ExecutableHarness.from_patch(patch, sandbox=LocalSandbox())
outcome = agent.run(task, harness=harness)   # your agent, now guarded
```

## When not to use it

- **No measurable outcome.** If you cannot compute a success/score per task there
  is nothing to optimize - the reward is the rerun delta, not a judge.
- **No failures.** If the agent already succeeds, there is no signal to learn from.
- **The model or prompt can change freely.** That may be simpler; Harnyx is for a
  fixed model where you want the runtime improved from evidence.
- **Same-batch only.** The reward is transductive (the same tasks before/after),
  exactly as in the paper - not a held-out generalization guarantee.
- **Integration cost.** You must implement an `Environment` (and usually a
  tool-calling `Policy`) for your domain.

## Nyvero example

Nyvero is never a dependency of Harnyx core. The adapter targets a documented
duck-typed contract (see [`docs/nyvero.md`](https://github.com/nilaymallikk/Harnyx/blob/main/docs/nyvero.md)):

```python
from harnyx.adapters.nyvero import NyveroAgentAdapter, NyveroHarnessAdapter

harnyx_agent = NyveroAgentAdapter(nyvero_agent, benchmark="nyvero")
harnyx_harness = NyveroHarnessAdapter(nyvero_harness)  # expose Nyvero's harness
result = harnyx_agent.run(task, harness=harnyx_harness)
```

## Reproduction instructions

```bash
# Deterministic local end-to-end (works in CI, no model)
harnyx run

# Full local reproduction: toy loop + A-F ablations + security smoke
python examples/reproduction/run_local_reproduction.py --run-root runs/reproduction
# Full paper reproduction (WebShop/ALFWorld/DBBench) is documented, not runnable
# here because benchmark assets, Qwen3.5 models, and 8xH800 are unavailable.
```

The honest reproduction status — expected paper numbers vs what was actually
observed on the available hardware — is in [`docs/research.md`](https://github.com/nilaymallikk/Harnyx/blob/main/docs/research.md).
No benchmark result in this repository is fabricated.

To reproduce the paper protocol on real benchmark runtimes, point
`HarnessR1BenchmarkAdapter` at the reference AgentBench runtimes (or any
harness-aware runner) and use the released reference task splits.

## Security model

Generated harness code is untrusted. Every candidate passes:

```text
parse -> schema validation -> AST policy -> sandbox smoke test -> execution
```

`harnyx.sandbox.policy` rejects imports, filesystem/network/subprocess access,
dynamic evaluation (`eval`/`exec`/`compile`), dunder/attribute escapes,
`getattr`/`setattr`/`globals`/`locals`, async/class/with/lambda/while/yield,
generators, raise, and benchmark-answer leakage (e.g. numbered ALFWorld
instances). Execution uses a restricted builtin set, a wall-clock timeout, and a
line budget; runtime failures degrade to *no intervention*. `SubprocessSandbox`
adds process isolation, a sanitized environment (no credentials), and
`RLIMIT_AS`/`RLIMIT_CPU`. See [`docs/sandbox.md`](https://github.com/nilaymallikk/Harnyx/blob/main/docs/sandbox.md).

No API key is ever hard-coded or logged; providers read them from environment
variables.

## Benchmarks and adapters

Harnyx ships a deterministic local evaluator and a benchmark adapter. The reference
paper's WebShop (500 tasks), ALFWorld (500 tasks), and DBBench (300 tasks) are
supported through `HarnessR1BenchmarkAdapter` but their runtimes are not bundled.
It is deliberately *not* a core dependency of Harnyx.

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

Apache-2.0. Harnyx is an independent implementation; see [`NOTICE`](https://github.com/nilaymallikk/Harnyx/blob/main/NOTICE) for
attribution of the Harness-R1 reference implementation, Life-Harness/AgentBench,
Relax, and the benchmark environments.
