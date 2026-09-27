from __future__ import annotations

import pytest

from harnyx.engineering.patch import HarnessPatch, extract_json_object, extract_patch
from harnyx.errors import PatchParseError, PatchValidationError

VALID_RESPONSE = """<think>
Recurring failures show a premature submit.
</think>
<patch>
{
  "benchmark": "toy",
  "description": "Block submit until verified.",
  "actions": [
    {"type": "add_code_hook", "hook": "on_before_action", "code": "def hook(ctx, nb):\\n    return None\\n"}
  ]
}
</patch>
"""


def test_parse_complete_think_patch_response() -> None:
    patch = extract_patch(VALID_RESPONSE, benchmark="toy", require_think=True)
    assert isinstance(patch, HarnessPatch)
    assert patch.benchmark == "toy"
    assert patch.hook_names == ("on_before_action",)
    assert "premature submit" in patch.think


def test_parse_prefilled_think_response() -> None:
    # A prefilled-think completion continues after the opened <think> and emits
    # its own closing </think>.
    prefilled = VALID_RESPONSE.replace("<think>\n", "", 1)
    assert not prefilled.startswith("<think>")
    patch = extract_patch(prefilled, benchmark="toy", prefill_think=True)
    assert patch.benchmark == "toy"


def test_parse_json_from_fenced_block_and_trailing_text() -> None:
    text = (
        "```json\n"
        '{"benchmark": "toy", "actions": [{"type": "add_code_hook", "hook": "on_init", '
        '"code": "def hook(ctx, nb):\\n    return None\\n"}]}\n'
        "```\n trailing noise"
    )
    patch = extract_patch(text)
    assert patch.hook_names == ("on_init",)


def test_parse_rejects_missing_think_when_required() -> None:
    with pytest.raises(PatchParseError):
        extract_patch('<patch>{"benchmark": "toy", "actions": []}</patch>', require_think=True)


def test_parse_rejects_malformed_json() -> None:
    with pytest.raises(PatchParseError):
        extract_patch("<patch>{not json</patch>")


def test_parse_rejects_empty_actions() -> None:
    with pytest.raises(PatchValidationError):
        extract_patch('<patch>{"benchmark": "toy", "actions": []}</patch>')


def test_parse_rejects_benchmark_mismatch() -> None:
    with pytest.raises(PatchValidationError):
        extract_patch(VALID_RESPONSE, benchmark="other")


def test_prefilled_response_must_not_reopen_think() -> None:
    with pytest.raises(PatchParseError):
        extract_patch("<think>again</think>" + VALID_RESPONSE, prefill_think=True)


def test_extract_json_object_prefers_last_patch_like_object() -> None:
    text = '{"benchmark": "a", "actions": []} then {"benchmark": "b", "actions": [1]}'
    obj = extract_json_object(text)
    assert obj["benchmark"] == "b"


def test_patch_to_dict_round_trip() -> None:
    patch = extract_patch(VALID_RESPONSE, benchmark="toy")
    restored = HarnessPatch.from_dict(patch.to_dict())
    assert restored.hooks == patch.hooks
    assert restored.benchmark == patch.benchmark
