# Writing a custom evaluator

An evaluator runs an agent over a task batch and returns an `EvaluationResult`
keyed by `Task.id` (so baseline and patched runs can be compared per task).

```python
from harnyx.core.result import EvaluationResult

class Evaluator(Protocol):
    name: str
    def evaluate(self, agent, tasks, *, harness=None, harness_version="harness-v0") -> EvaluationResult: ...
```

## Requirements

1. **Same task identities** before and after a patch. The reward is transductive;
   it is only meaningful on a matched batch.
2. **Deterministic rewards** where possible (temperature 0, fixed seeds).
3. **No hidden judge.** The reward is the native task outcome (success or shaped
   score), not an LLM rating.

## Example

```python
class MyEvaluator:
    name = "my-evaluator"
    def evaluate(self, agent, tasks, *, harness=None, harness_version="harness-v0"):
        result = EvaluationResult(benchmark="mybench", harness_version=harness_version,
                                  metadata={"harness_noop": harness is None or harness.is_noop})
        for task in tasks:
            outcome = agent.run(task, harness=harness)
            result.rewards[task.id] = outcome.reward
            result.successes[task.id] = outcome.success
            result.trajectories[task.id] = outcome.trajectory
        return result
```

## External benchmarks

Use `HarnessR1BenchmarkAdapter` for a harness-aware external runner:

```python
from harnyx.evaluation.harness_r1 import HarnessR1BenchmarkAdapter

evaluator = HarnessR1BenchmarkAdapter(runner, benchmark="webshop")
```

The runner owns the target agent; the adapter normalizes `{reward, success,
trajectory, error}` into an `EvaluationResult`.

## Incomplete evaluations

If any task errors, `EvaluationResult.errors` is non-empty. With the default
`OutcomeReward(require_complete=True)`, the candidate scores `0` rather than a
partially-optimistic reward. This mirrors the reference rule that invalid or
incomplete evaluations receive zero reward.
