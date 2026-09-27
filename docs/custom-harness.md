# Writing a custom harness

A harness implements four methods:

```python
class Harness(Protocol):
    name: str
    def on_init(self, ctx, nb) -> HookEffect | None: ...
    def make_pre_hint(self, ctx, nb) -> HookEffect | None: ...
    def on_before_action(self, ctx, nb) -> HookEffect | None: ...
    def on_post_step(self, ctx, nb) -> HookEffect | None: ...
```

The standard implementation is `ExecutableHarness`, built from a patch:

```python
from nova import ExecutableHarness, LocalSandbox, HarnessPatch

patch = HarnessPatch.from_dict({
    "benchmark": "mybench",
    "description": "block premature commits",
    "actions": [{
        "type": "add_code_hook",
        "hook": "on_before_action",
        "code": (
            "def hook(ctx, nb):\n"
            "    state = ctx.get('state') or {}\n"
            "    action = ctx.get('action') or {}\n"
            "    if str(action.get('name')) == 'submit' and not state.get('verified'):\n"
            "        return {'kind': 'block_and_prompt', 'message': 'Verify before submitting.'}\n"
            "    return None\n"
        ),
    }],
})

harness = ExecutableHarness.from_patch(patch, sandbox=LocalSandbox())
```

## Context available to hooks

`HookContext.to_dict()` exposes:

```python
{
  "benchmark": str, "observation": str, "step": int, "max_step": int,
  "remaining_steps": int, "task": {...}, "state": {...}, "predicates": {...},
  "action": {...} | None, "admissible": [...], **benchmark_namespace
}
```

## Return contracts

| Hook | Return |
|---|---|
| `on_init` | `{"skills": [{"text": ...}], "tool_hint": str}` or `None` |
| `make_pre_hint` | `{"message": str}` or `None` |
| `on_before_action` | `{"kind": "block_and_prompt" | "rewrite_action" | "force_action", "message"?: str, "action"?: str}` or `None` |
| `on_post_step` | `{"kind": "inject_hint" | "force_action", "message"?: str, "action"?: str}` or `None` |

Malformed returns normalize to `None`. An empty action on a force/rewrite
degrades to no effect.

## Static custom harness

You can subclass `BaseHarness` directly for a hand-written harness that is not
generated. It is still subject to the same effect normalization if you route it
through the sandbox; otherwise it is trusted code.

## Constraints

- Hook bodies are ordinary Python but a strict subset: no imports, no I/O, no
  dunder access, no dynamic evaluation, no `while`, no `raise`, no classes, no
  lambdas, at most 5 scalar-default helpers.
- At most one hook per lifecycle position per patch.
- Hooks must be deterministic and side-effect the notebook `nb`, not the world.
