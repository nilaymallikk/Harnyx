from __future__ import annotations

from pathlib import Path

from nova.engineering.patch import extract_patch
from nova.engineering.validation import PatchValidator

REPO = Path(__file__).resolve().parents[1]


def test_example_engineer_response_parses_and_validates() -> None:
    text = (REPO / "examples" / "patches" / "example_patch.txt").read_text(encoding="utf-8")
    patch = extract_patch(text, benchmark="verify", require_think=True, prefill_think=True)
    assert patch.benchmark == "verify"
    assert patch.hook_names == ("on_before_action",)
    result = PatchValidator().validate(patch)
    assert result.ok, result.errors


def test_plugin_example_build_shape() -> None:
    from examples.plugins.verify_agent import build

    payload = build()
    assert payload["benchmark"] == "verify"
    assert len(payload["tasks"]) == 2
    assert payload["engineer"] is not None
    assert hasattr(payload["agent"], "run")
