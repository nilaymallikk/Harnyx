from __future__ import annotations

import pytest

from harnyx.engineering.patch import CodeHook, HarnessPatch
from harnyx.engineering.validation import PatchValidator

SAFE = "def hook(ctx, nb):\n    return None\n"


def make_patch(*hooks: tuple[str, str], benchmark: str = "toy") -> HarnessPatch:
    return HarnessPatch(benchmark=benchmark, hooks=tuple(CodeHook(hook=h, code=c) for h, c in hooks))


def test_valid_patch_passes() -> None:
    result = PatchValidator().validate(make_patch(("on_init", SAFE), ("make_pre_hint", SAFE)))
    assert result.ok is True
    assert result.patch is not None
    assert result.errors == []


def test_invalid_syntax_rejected() -> None:
    result = PatchValidator().validate(make_patch(("on_init", "def hook(ctx, nb):\n    return (") ))
    assert result.ok is False
    assert any("syntax" in error for error in result.errors)


def test_empty_patch_rejected() -> None:
    result = PatchValidator().validate(HarnessPatch(benchmark="toy", hooks=()))
    assert result.ok is False
    assert any("no actions" in error for error in result.errors)


def test_duplicate_hook_rejected() -> None:
    result = PatchValidator().validate(make_patch(("on_init", SAFE), ("on_init", SAFE)))
    assert result.ok is False
    assert any("at most once" in error for error in result.errors)


def test_too_many_hooks_rejected() -> None:
    hooks = tuple((name, SAFE) for name in ("on_init", "make_pre_hint", "on_before_action", "on_post_step"))
    assert PatchValidator().validate(make_patch(*hooks)).ok
    result = PatchValidator(max_hooks=2).validate(make_patch(*hooks))
    assert result.ok is False


@pytest.mark.parametrize(
    "source",
    [
        "import os\ndef hook(ctx, nb):\n    return None\n",
        "def hook(ctx, nb):\n    return open('secret')\n",
        "def hook(ctx, nb):\n    return __import__('subprocess')\n",
        "def hook(ctx, nb):\n    return eval('1+1')\n",
        "def hook(ctx, nb):\n    return ().__class__.__bases__\n",
    ],
)
def test_forbidden_hook_sources_rejected(source: str) -> None:
    result = PatchValidator().validate(make_patch(("on_before_action", source)))
    assert result.ok is False
    assert result.errors


def test_wrong_action_type_rejected() -> None:
    result = PatchValidator().validate(
        {"benchmark": "toy", "actions": [{"type": "set_config", "target": "x", "value": 1}]}
    )
    assert result.ok is False


def test_raise_if_invalid_raises() -> None:
    from harnyx.errors import PatchValidationError

    result = PatchValidator().validate(make_patch())
    with pytest.raises(PatchValidationError):
        result.raise_if_invalid()
