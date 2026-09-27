# Quickstart

## 1. Install

```bash
pip install -e .
```

## 2. Run the deterministic toy demo

```bash
harnyx run --run-dir runs --candidates 3
```

Expected output:

```text
Harnyx toy demo
  research run directory : runs/<timestamp>
  baseline success       : 0/1  (mean reward 0.000)
  patched success        : 1/1  (mean reward 1.000)
  engineer reward        : +1.000
  accepted versions      : harness-v1
```

The scenario: a frozen policy submits before verifying. The engineer's pre-action
guard blocks the premature submit; the policy verifies, then submits. Baseline
reward 0, patched reward 1, engineer reward +1.

Inspect the run:

```bash
ls runs/<timestamp>
cat runs/<timestamp>/report.json
cat runs/<timestamp>/accepted_patch.json
```

## 3. Use your own agent

See [`custom-agent.md`](custom-agent.md). Then:

```python
from harnyx import HarnessOptimizer, LocalEvaluator
from harnyx.optimization.optimizer import OptimizationConfig

result = HarnessOptimizer(
    agent, engineer,
    evaluator=LocalEvaluator(benchmark="mybench"),
    benchmark="mybench",
    config=OptimizationConfig(candidates=8, iterations=3),
    run_dir="runs",
).optimize(tasks)
```

## 4. Configure an LLM engineer

```yaml
# configs/harnyx.example.yaml
engineer:
  provider: openai_compatible
  model: Qwen3.5-9B-engineer
  base_url: http://localhost:8000/v1
  api_key_env: HARNYX_ENGINEER_API_KEY
optimization:
  candidates: 8
  iterations: 5
sandbox:
  backend: local
  hook_time_budget_ms: 50
evaluation:
  benchmark: mybench
```

```bash
export HARNYX_ENGINEER_API_KEY=...
harnyx optimize --config configs/harnyx.example.yaml --plugin mypkg.plugin:build
```

## 5. Inspect trajectories

```bash
harnyx inspect-trajectory runs/<timestamp>/... --task-id task-3
```

## 6. Run ablations on the toy benchmark

```python
from harnyx.demo.toy import build_toy_agent, build_toy_tasks
from harnyx.experiments.ablations import run_all_ablations

for r in run_all_ablations(build_toy_agent(), build_toy_tasks(), benchmark="toy", run_root="runs/ablations"):
    print(r.key, r.name, f"{r.reward:+.3f}")
```
