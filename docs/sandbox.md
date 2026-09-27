# Sandbox and security model

Generated harness code is **untrusted**. Harnyx never trusts the engineer, the
provider, or the model output.

## Pipeline

```text
model output
  └─ parse <think>/<patch>                     harnyx.engineering.patch
       └─ schema validation                    harnyx.engineering.validation
            └─ AST policy                      harnyx.sandbox.policy
                 └─ compile + smoke test       harnyx.sandbox.runner / isolation
                      └─ execute in rollout    LocalSandbox or SubprocessSandbox
```

Each gate is independent; a failure at any gate yields *no intervention* or a
rejected candidate, never partial execution.

## AST policy (`harnyx.sandbox.policy`)

Rejected constructs include:

- **Imports / filesystem / network / subprocess:** `import`, `from ... import`,
  `open`, `__import__`, and every name/attribute in the forbidden set.
- **Dynamic execution:** `eval`, `exec`, `compile`, `globals`, `locals`,
  `getattr`, `setattr`, `delattr`, `vars`, `dir`, `input`, `breakpoint`.
- **Escapes:** any attribute or name starting with `_` (blocks
  `().__class__.__bases__`, `__builtins__`, etc.).
- **Control-flow hazards:** `while`, `class`, `lambda`, `with`, `async`,
  `yield`, nested functions, `raise`, `delete`, `global`, `nonlocal`.
- **Exception tricks:** `try` must catch exactly `except Exception` with no
  `else`/`finally`.
- **Leakage:** string literals that hard-code numbered ALFWorld instance actions
  (e.g. `go to sinkbasin 1`), plus structural caps on source length, AST nodes,
  helper count, and string length.

The hook signature is exactly `hook(ctx, nb)`; helpers may only use simple scalar
defaults and no annotations or decorators.

## Execution budgets (`harnyx.sandbox.limits.SandboxLimits`)

| Limit | Default | Meaning |
|---|---|---|
| `time_budget_s` | 0.05 | `SIGALRM` wall budget per hook call |
| `line_budget` | 2000 | executed-line budget via `sys.settrace` |
| `max_source_chars` | 8000 | hook source cap |
| `max_ast_nodes` | 1200 | AST size cap |
| `max_helper_functions` | 5 | helper cap |
| `max_return_text_chars` | 900 | returned message/action cap |
| `memory_mb` / `cpu_seconds` | 256 / 2 | subprocess `RLIMIT_AS`/`RLIMIT_CPU` |
| `wall_timeout_s` | 5.0 | subprocess wall timeout |

Runtime exceptions and timeouts in `LocalSandbox` degrade to `None`. The
subprocess sandbox raises `SandboxTimeout`/`SandboxError` so the caller can mark
a candidate invalid.

## Restricted builtins

Hooks only see:
`abs, all, any, bool, dict, enumerate, Exception, filter, float, int, isinstance,
len, list, map, max, min, range, reversed, round, set, sorted, str, sum, tuple,
zip`, plus `math`, `re`, and `SequenceMatcher` in the namespace.

## Process isolation (`harnyx.sandbox.isolation.SubprocessSandbox`)

- fresh interpreter per call,
- sanitized environment (`PATH`, `PYTHONPATH`, `LANG`, hash seed only — no API
  keys or credentials),
- `RLIMIT_AS` and `RLIMIT_CPU` where the platform supports them,
- wall-clock timeout,
- notebook mutations returned to the caller.

Use it as a pre-flight smoke test for engineer output, or as the runtime
backend when stronger isolation matters more than throughput.

## What this does *not* protect against

- A hook is still Python in the same address space under `LocalSandbox`; the
  AST policy is the boundary, exactly as in the reference implementation. Use
  `SubprocessSandbox` when you need an OS-level boundary.
- Denial of service is capped by budgets, not eliminated.
- The engine (the LLM) is not sandboxed; only its generated harness code is.

## Tests

Adversarial cases live in `tests/test_sandbox.py` and
`tests/test_patch_validator.py`: forbidden import, filesystem access,
subprocess, network, dynamic execution, dunder escape, runtime exception,
timeout, line budget, and ALFWorld leakage.
