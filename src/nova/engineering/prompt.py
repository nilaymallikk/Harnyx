"""Harness-engineer prompt construction (paper Appendix A).

The system/input/response protocol is reproduced here so SFT data, online GRPO
rollouts, and evaluation all share one static protocol. Benchmark-specific
sections expose runtime evidence, never hidden task answers.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

SYSTEM_INSTRUCTION = (
    "You are a Harness-R1 harness engineer. Given a batch of failed "
    "{benchmark} rollout traces, analyze recurring failures, then produce one "
    "reusable {benchmark} code-hook harness patch as a single JSON object inside "
    "a <patch>...</patch> block."
)

INPUT_PREAMBLE = """You will edit only the reusable {benchmark} harness, not task answers.

The chat template has already opened the assistant thinking block. Continue
concise recurring-failure reasoning, close it with </think>, then output
exactly one <patch> block.

After your reasoning, output exactly one <patch> block containing a single
JSON patch object with top-level keys benchmark, description, and actions.
Output nothing after the </patch> block.

Patch JSON top-level contract:
- benchmark: exactly "{benchmark}"
- description: a short, general description
- actions: a non-empty array of ADD_CODE_HOOK objects

ADD_CODE_HOOK ::= {{
  "type": "add_code_hook",
  "hook": "on_init" | "make_pre_hint" |
          "on_before_action" | "on_post_step",
  "code": "<PYTHON SOURCE DEFINING hook(ctx, nb)>"
}}

Hook return contracts:
- on_init -> skills/tool_hint or None
- make_pre_hint -> message or None
- on_before_action -> block_and_prompt, rewrite_action, or force_action
- on_post_step -> inject_hint or, where supported, force_action

General rules:
- Infer reusable interventions from observable recurring failures.
- Do not encode task-specific answers, indices, or identifiers.
- Define hook(ctx, nb) without imports or global state.
- Put all JSON inside the <patch> block; output nothing after </patch>."""

BENCHMARK_CONTEXT: dict[str, str] = {
    "webshop": """[WebShop]
benchmark = "webshop"
Active action type: add_code_hook only.
Runtime context:
  action: tool, value, value_normalized, final_action
  state: page_type, clickables, remaining_steps, current_price, price_max,
         repeated-click/search and product-stall counters
  task: task_type, required_color, required_size, required_material
  webshop: search_queries, visited products, current product,
           selected attributes, available attribute options
  predicates: required_options_unselected, product_price_over_budget,
              repeated click/search, stalled product page,
              buy-now availability, action admissibility
Use pre-action mediation only for narrow, observable mistakes such as an
over-budget purchase, an unselected required option, or a repeated action.
Do not hard-code product IDs, titles, ASINs, task indices, or answers.""",
    "alfworld": """[ALFWorld]
benchmark = "alfworld"
Actions may contain two to four add_code_hook entries; use each hook at most
once and omit hooks not supported by recurring evidence.
Runtime context:
  action: raw, normalized, final_action, in_admissible
  state: repeated observation/action, invalid-action and remaining-step signals
  task: task_type, target_type, destination_type
  world: current location, inventory, object locations, visited locations,
         target facts, and placement facts
  admissible: actions currently accepted by the environment
Maintain reusable task-stage state from observations and completed actions.
Any exact mediated action must be selected from admissible actions. Do not
copy numbered object/location instances or complete task solutions.""",
    "dbbench": """[DBBench]
benchmark = "dbbench"
Actions may contain two to four add_code_hook entries; use each hook at most
once and omit hooks not supported by recurring evidence.
Runtime context:
  action: execute_sql or commit_final_answer, query, submitted answers
  state: SQL count/history, last result/error, error/empty/loop streaks,
         mutation status, candidate answer shape, remaining rounds
  task: task_type, answer_shape, target_table, description
  dbbench: SQL history, discovered columns, database response, round
  predicates: premature/empty commit, mutation not attempted, SQL error,
              empty result, unknown column, syntax error, repeated SQL,
              candidate answer available, remaining rounds low
DBBench hooks may inject guidance or block a pending action. They do not
rewrite SQL or force a commit. Do not hard-code cell values, exact answers,
task indices, or ground-truth SQL.""",
}

RESPONSE_TEMPLATE = """<think>
<concise analysis of recurring failures, the proposed reusable intervention,
and the regression risk>
</think>
<patch>
{{
  "benchmark": "{benchmark}",
  "description": "<GENERAL PATCH DESCRIPTION>",
  "actions": [
    {{
      "type": "add_code_hook",
      "hook": "<SELECTED LIFECYCLE POSITION>",
      "code": "def hook(ctx, nb):\\n    ...\\n    return <STRUCTURED EFFECT>"
    }}
  ]
}}
</patch>"""


def benchmark_context(benchmark: str) -> str:
    """Return the benchmark-specific prompt section (generic fallback if unknown)."""
    key = (benchmark or "").strip().lower()
    if key in BENCHMARK_CONTEXT:
        return BENCHMARK_CONTEXT[key]
    return (
        f"[{benchmark or 'generic'}]\n"
        f'benchmark = "{benchmark}"\n'
        "Active action type: add_code_hook only.\n"
        "Hook return contracts are the same as the general contract above."
    )


def packet_prompt_text(packet: Any) -> str:
    """Render a failure packet to its prompt text."""
    render = getattr(packet, "to_prompt_text", None)
    if callable(render):
        return str(render())
    return str(packet)


def build_engineer_messages(
    packet: Any,
    *,
    benchmark: str | None = None,
    include_response_template: bool = False,
) -> list[dict[str, str]]:
    """Build the ordered system/user messages for the harness engineer.

    Args:
        packet: A failure packet (anything exposing ``to_prompt_text()``).
        benchmark: Benchmark id; falls back to ``packet.benchmark``.
        include_response_template: Append Figure A.3 as an in-context reminder.
    """
    bench = benchmark or getattr(packet, "benchmark", "") or "generic"
    system = SYSTEM_INSTRUCTION.format(benchmark=bench)
    sections = [
        INPUT_PREAMBLE.format(benchmark=bench),
        benchmark_context(bench),
        "Observed no-harness rollout evidence:\n\n" + packet_prompt_text(packet),
    ]
    if include_response_template:
        sections.append("Expected response format:\n\n" + RESPONSE_TEMPLATE.format(benchmark=bench))
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": "\n\n".join(sections)},
    ]


def packet_metadata(packet: Any) -> Mapping[str, Any]:
    """Best-effort access to packet metadata for provenance checks."""
    return getattr(packet, "metadata", {}) or {}
