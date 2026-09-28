# Using Harnyx with Nyvero

[Nyvero](https://github.com/NilLab-agi/Nyvero) is a minimal AI coding-agent
harness: one loop that calls an OpenAI-compatible model, runs each tool call
through one shared permission-checked executor, appends the results, and repeats.

Harnyx wraps that loop so it can mine Nyvero's failures and install an
executable harness patch — **without editing Nyvero and without touching the
model**. Nyvero is never a dependency of Harnyx; the adapter loads it lazily.

## Where the hooks attach

From `nyvero/agent.py`:

| Nyvero loop point | Harnyx hook | What the patch can do |
|---|---|---|
| after `Context()` init | `on_init` | add reusable skills / tool hints to the system context |
| before `stream_llm(messages, tools)` | `make_pre_hint` | inject a state-conditioned hint |
| before `execute_tool(name, arguments)` | `on_before_action` | **block** a call, **rewrite** its arguments, or **force** a different one |
| after `execute_tool` returns | `on_post_step` | inject a recovery hint after an error |

The context a hook sees (`ctx`) includes the conversation, the last tool result,
`step` / `remaining_steps`, the pending `action`, and derived state such as
`error_streak` and the list of admissible tools, under `state` and `predicates`.

## The one thing you must provide: an objective reward

A coding agent has no built-in success signal, so you must say what "done" means.
Harnyx's reward is the realized **rerun delta**, never a judge, so the check must
be objective. The usual choice is the repository's tests:

```python
outcome = per_task_check({
    "fix-parser-edge-case": "uv run pytest tests/test_parser.py -q",
    "add-missing-validation": "uv run pytest tests/test_config.py -q",
})
# success = exit code 0, reward = 1.0 or 0.0
```

`workspace_check(command)` is the single-task form. Both run *your* trusted
command, never model output.

## Run the optimization

```bash
python examples/nyvero/optimize_nyvero.py \
    --nyvero-repo ~/Nyvero \
    --workspace ~/myproject \
    --engineer-base-url https://api.deepseek.com \
    --engineer-model deepseek-chat
```

In code:

```python
from harnyx import Task, HarnessOptimizer, LLMHarnessEngineer, LocalEvaluator
from harnyx.adapters.nyvero import NyveroAgentAdapter, NyveroBackend, per_task_check
from harnyx.optimization.optimizer import OptimizationConfig

backend = NyveroBackend.from_installation(repo="~/Nyvero", auto_approve=True)

agent = NyveroAgentAdapter(
    backend,
    benchmark="nyvero",
    workspace="~/myproject",
    max_steps=30,
    outcome=per_task_check(checks, timeout=900.0),
)

result = HarnessOptimizer(
    agent,
    LLMHarnessEngineer(provider, benchmark="nyvero"),
    evaluator=LocalEvaluator(benchmark="nyvero"),
    benchmark="nyvero",
    config=OptimizationConfig(candidates=8, iterations=3),
    run_dir="runs/nyvero",
).optimize(tasks)          # tasks: list[Task]
```

What happens: baseline run → each task's check → failures mined into a packet →
the engineer proposes `K` patches → each is validated and sandboxed → Nyvero
reruns the **same tasks** with the patch → the patch is kept only if the check
improves and nothing regresses. Results land in `runs/nyvero/<ts>/`.

## Non-interactive / safety notes

- `auto_approve=True` replaces Nyvero's interactive `ui.confirm` with an
  automatic yes so the loop can run headless. **`DENY` rules still apply**
  (destructive commands), and bash still runs inside Nyvero's OS sandbox
  (bubblewrap/seatbelt) where available.
- Run the optimization against a disposable checkout or container. The agent
  edits files in `--workspace`; the check commands run there too.
- The engineer endpoint only writes harness patches. Those patches are validated
  and executed in Harnyx's sandbox (AST policy: no imports, I/O, network), not
  given to Nyvero as free code.

## Installing an accepted patch into a live Nyvero session

Load the accepted patch and call the four hooks in Nyvero's loop:

```python
import json
from harnyx import ExecutableHarness, HarnessPatch, LocalSandbox
from harnyx.adapters.nyvero import build_hook_bridge

patch = HarnessPatch.from_dict(json.load(open("runs/nyvero/<ts>/accepted_patch.json"))["patch"])
harness = ExecutableHarness.from_patch(patch, sandbox=LocalSandbox())
bridge = build_hook_bridge(harness)      # {"on_init", "make_pre_hint", "on_before_action", "on_post_step"}
```

Then in `nyvero/agent.py`:

```python
# before the model call
effect = bridge["make_pre_hint"]({"observation": "", "state": state, "step": step})
if effect and effect.get("message"):
    context.messages.append({"role": "user", "content": effect["message"]})

# around execute_tool
effect = bridge["on_before_action"]({"action": {"name": name, "arguments": arguments}, "state": state})
if effect and effect.get("kind") == "block_and_prompt":
    tool_result = "[harness] " + effect.get("message", "blocked")
else:
    tool_result = execute_tool(name, arguments)

# after the result
effect = bridge["on_post_step"]({"action": {"name": name, "arguments": arguments},
                                 "observation": tool_result, "state": state})
if effect and effect.get("kind") == "inject_hint":
    context.messages.append({"role": "user", "content": effect["message"]})
```

Keep this behind a flag so Nyvero still runs unpatched by default.

## Limitations

- Nyvero's own permission layer remains the authority; a Harnyx guard adds to it,
  it does not replace it.
- `force_action` / `rewrite_action` map a hook action string onto the pending
  call's first string argument (`command` for `bash`, `path` for file tools).
  Pass `apply_effect=` to `NyveroAgentAdapter` for structured rewrites.
- Subagents run Nyvero's same loop with a fresh context; this adapter covers the
  main loop.
