#!/usr/bin/env python3
"""Runnable local reproduction harness.

This is the closest end-to-end reproduction possible without benchmark runtimes,
model checkpoints, or GPUs. It runs:

1. the deterministic toy Harness-R1 loop (baseline -> patch -> rerun -> reward),
2. the A-F ablation arms on the toy benchmark,
3. the adversarial sandbox security suite (as a quick sanity summary),

and writes a machine-readable report. It does NOT claim paper-level results; see
``docs/research.md`` for the honest expected-vs-observed table.

Usage::

    python examples/reproduction/run_local_reproduction.py --run-root runs/reproduction
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
if str(REPO / "src") not in sys.path:
    sys.path.insert(0, str(REPO / "src"))

from harnyx.demo.toy import TOY_BENCHMARK, build_toy_agent, build_toy_tasks, run_demo  # noqa: E402
from harnyx.experiments.ablations import ABLATIONS, run_all_ablations  # noqa: E402
from harnyx.sandbox.runner import LocalSandbox  # noqa: E402


def security_smoke() -> dict[str, bool]:
    """Confirm the AST policy rejects the headline attack classes."""
    attacks = {
        "import": "import os\ndef hook(ctx, nb):\n    return None\n",
        "filesystem": "def hook(ctx, nb):\n    return open('/etc/passwd')\n",
        "subprocess": "def hook(ctx, nb):\n    return __import__('subprocess')\n",
        "dynamic_exec": "def hook(ctx, nb):\n    return eval('1')\n",
        "dunder_escape": "def hook(ctx, nb):\n    return ().__class__.__bases__\n",
    }
    sandbox = LocalSandbox()
    out: dict[str, bool] = {}
    for name, source in attacks.items():
        try:
            sandbox.compile(source)
            out[name] = False
        except Exception:
            out[name] = True
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", default="runs/reproduction")
    args = parser.parse_args()
    root = Path(args.run_root)
    root.mkdir(parents=True, exist_ok=True)

    toy = run_demo(root / "toy", candidates=3, iterations=1)
    ablations = run_all_ablations(
        build_toy_agent(), build_toy_tasks(), benchmark=TOY_BENCHMARK, run_root=root / "ablations"
    )
    security = security_smoke()

    report = {
        "toy": toy,
        "ablations": [result.to_dict() for result in ablations],
        "security": security,
        "note": (
            "Local deterministic reproduction only. WebShop/ALFWorld/DBBench paper "
            "numbers are not reproduced here; see docs/research.md."
        ),
    }
    (root / "reproduction.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    print("Harness-R1 local reproduction")
    print(f"  toy: baseline={toy['baseline_reward']:.3f} patched={toy['patched_reward']:.3f} "
          f"reward={toy['reward']:+.3f} accepted={toy['accepted']}")
    print("  ablations:")
    for result in ablations:
        print(f"    {result.key} {ABLATIONS[result.key].name:<20} reward={result.reward:+.3f}")
    print("  security:", json.dumps(security, sort_keys=True))
    print(f"  wrote {root / 'reproduction.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
