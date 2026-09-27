# Concepts

## Agent
A frozen decision policy plus its base runtime. In Harnyx an agent is anything
implementing `run(task, harness=None, recorder=None) -> AgentResult`. The base
runtime is `harnyx.core.agent.HarnessedAgent`, which drives the four lifecycle
hooks around a `Policy` and an `Environment`.

## Harness
The editable runtime surrounding the policy. Harnyx harnesses expose exactly four
methods and can only return structured `HookEffect`s. `BaseHarness` is the no-op
`harness-v0`. `ExecutableHarness` is built from a validated patch.

## Trajectory
Observable execution information only: observations, proposed/executed actions,
tool results, harness effects, errors, timestamps, state snapshots, and the final
result. Chain-of-thought is never required or recorded.

## Failure packet
The minimum evidence the engineer needs: only failed episodes, their task
constraints, bounded action/observation excerpts, recurring runtime signals, and
the baseline outcome. It is deterministic and serializable.

## Harness patch
A single JSON object `{benchmark, description, actions}`, where every action is
`{"type": "add_code_hook", "hook": ..., "code": "def hook(ctx, nb): ..."}`.
At most one hook per lifecycle position.

## Outcome reward
`Δ = mean(patched reward) − mean(baseline reward)` over the same task identities.
Zero for invalid, no-op, or incomplete evaluation. No LLM judge.

## Candidate selection
Among `K` candidates, pick the highest-reward one that does not regress a
previously solved task. Regression protection can be relaxed explicitly.

## Harness version
Every accepted edit is stored as `harness-v{n}` with parent, patch, tasks,
scores, reward, timestamp, model, and configuration — enabling inspection and
rollback.
