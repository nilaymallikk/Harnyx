# Examples

## `plugins/verify_agent.py`

A self-contained Harnyx agent/tasks/engineer plugin for the CLI. The frozen policy
reports before inspecting; a pre-action guard blocks the premature report.

```bash
harnyx evaluate --plugin examples.plugins.verify_agent:build
harnyx optimize --plugin examples.plugins.verify_agent:build --run-dir runs/verify
```

Expected: baseline success `0/2`, patched success `2/2`, engineer reward `+1.000`.

## Example engineer response

`patches/example_patch.txt` is a complete `<think>...</think><patch>...</patch>`
engineer response you can feed through the parser:

```python
from harnyx.engineering.patch import extract_patch
text = open("examples/patches/example_patch.txt").read()
patch = extract_patch(text, benchmark="verify", require_think=True, prefill_think=True)
```

## Run directories

Each `harnyx run` / `harnyx optimize` writes a structured run directory (config,
baseline, failures, candidates, rewards, accepted patch, report). See
`docs/concepts.md`.
