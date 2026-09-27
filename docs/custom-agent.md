# Writing a custom agent

An agent is anything with:

```python
class Agent(Protocol):
    name: str
    def run(self, task, harness=None, recorder=None) -> AgentResult: ...
```

## Option 1 — the generic runtime

Supply a `Policy` and an `Environment`. Harnyx's `HarnessedAgent` drives the loop
and calls the four harness hooks.

```python
from harnyx import Action, Task, HarnessedAgent, StepResult

class MyEnv:
    name = "myenv"
    def reset(self, task: Task) -> str: ...
    def step(self, action: Action) -> StepResult: ...
    def state(self) -> dict: ...
    def predicates(self) -> dict: ...
    def success(self) -> bool: ...
    def episode_reward(self) -> float: ...
    def admissible_actions(self) -> list[str]: ...

class MyPolicy:
    name = "my-policy"
    def act(self, messages, *, step, admissible) -> Action: ...

agent = HarnessedAgent(MyPolicy(), MyEnv(), benchmark="mybench", max_steps=20)
```

Optional environment hooks:
- `predicates() -> dict` — exposed to hooks as `ctx["predicates"]`.
- `context_extra() -> dict` — merged into `ctx` at the top level (use for a
  benchmark namespace such as `{"mybench": {...}}`).

## Option 2 — wrap an existing runner

```python
from harnyx import AgentResult, TrajectoryRecorder

class MyAgent:
    name = "wrapped"
    def run(self, task, harness=None, recorder=None):
        recorder = recorder or TrajectoryRecorder(task)
        observation = self.reset(task)
        nb = {}
        ctx = ...  # build a HookContext
        effect = harness.on_init(ctx, nb) if harness else None
        ...
        trajectory = recorder.finish(reward=r, success=s, status="completed")
        return AgentResult(task_id=task.id, success=s, reward=r, trajectory=trajectory)
```

If you invoke the hooks yourself, honour the contract: `on_before_action` may
return `block_and_prompt` (re-prompt), `rewrite_action`/`force_action` (replace
the pending action), and `on_post_step` may return `inject_hint` or
`force_action`. Never let a hook call the environment directly.

## Option 3 — benchmark adapter

```python
from harnyx.evaluation.harness_r1 import HarnessR1BenchmarkAdapter

evaluator = HarnessR1BenchmarkAdapter(my_runner, benchmark="webshop")
```

`my_runner.run(task, harness)` must install the harness hooks at the four
lifecycle points and return `{reward, success, trajectory?, error?}`.
