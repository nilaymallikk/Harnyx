# Nyvero adapter

Nyvero is **never** a dependency of NOVA core. `nova.adapters.nyvero` translates
between NOVA's `Agent`/`Harness` interfaces and a small, documented duck-typed
Nyvero contract. Nothing in the module imports Nyvero.

## Adapter contract

A Nyvero **agent** is any object exposing:

```python
name: str
def rollout(instruction: str, *, hooks: Mapping[str, Hook], **kwargs) -> NyveroRollout
```

where `Hook = Callable[[Mapping[str, Any]], Mapping[str, Any] | None]`, keyed by
the four lifecycle names. `NyveroRollout` exposes `success`, `reward`, `steps`,
and `status`. Each step exposes `observation`, `action`, `tool_result`, `error`,
`state` as attributes or mapping keys.

A Nyvero **harness** is any object exposing `on_init`, `make_pre_hint`,
`on_before_action`, and `on_post_step`, each accepting a context mapping and
returning a raw effect mapping.

## Usage

```python
from nova.adapters.nyvero import NyveroAgentAdapter, NyveroHarnessAdapter

# Translate a Nyvero agent into a NOVA Agent.
nova_agent = NyveroAgentAdapter(nyvero_agent, benchmark="nyvero")

# Expose a Nyvero harness through NOVA's controlled harness API.
nova_harness = NyveroHarnessAdapter(nyvero_harness)

result = nova_agent.run(task, harness=nova_harness)
```

Or let NOVA install a *generated* patch on a Nyvero agent:

```python
from nova import ExecutableHarness, LocalSandbox, HarnessOptimizer

harness = ExecutableHarness.from_patch(patch, sandbox=LocalSandbox())
HarnessOptimizer(nova_agent, engineer, benchmark="nyvero", run_dir="runs").optimize(tasks)
```

`NyveroAgentAdapter` builds the hook bridge (`build_hook_bridge`) so NOVA harness
effects reach Nyvero as raw mappings, and normalizes the returned rollout into a
NOVA `Trajectory` and `AgentResult`.

## Effect normalization

`NyveroHarnessAdapter` runs Nyvero's raw returns through the same
`normalize_effect` as the sandbox, so an invalid effect from a Nyvero harness
degrades to no intervention instead of crashing an episode.

## Limitation

The exact Nyvero release is not vendored, so this adapter targets the contract
above. Point it at a Nyvero runtime implementing that contract; if your Nyvero
exposes different method names, write a thin shim that maps them to
`on_init`/`make_pre_hint`/`on_before_action`/`on_post_step` and
`rollout(instruction, hooks=...)`. The conformance test in
`tests/test_adapters.py` uses a fake conforming runtime.
