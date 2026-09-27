"""JSON helpers and shared type aliases.

NOVA persists research artifacts (trajectories, failure packets, candidates,
rewards, run reports). Serialization must be deterministic so two runs with the
same inputs produce byte-identical files.
"""

from __future__ import annotations

import json
import os
import tempfile
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any, TypeAlias

JSONScalar: TypeAlias = str | int | float | bool | None
JSONValue: TypeAlias = JSONScalar | list["JSONValue"] | dict[str, "JSONValue"]
JSONObject: TypeAlias = dict[str, JSONValue]

# Notebook state shared between harness hooks across an episode.
Notebook: TypeAlias = dict[str, Any]


def to_jsonable(value: Any) -> Any:
    """Recursively convert dataclasses/mappings into JSON-serializable values."""
    from dataclasses import asdict, is_dataclass

    if is_dataclass(value) and not isinstance(value, type):
        return to_jsonable(asdict(value))
    if isinstance(value, Mapping):
        return {str(k): to_jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set, frozenset)):
        return [to_jsonable(v) for v in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def canonical_json(value: Any) -> str:
    """Serialize ``value`` deterministically (sorted keys, compact separators)."""
    return json.dumps(to_jsonable(value), sort_keys=True, ensure_ascii=False, separators=(",", ":"))


def write_json(path: str | os.PathLike[str], value: Any) -> Path:
    """Atomically write ``value`` as pretty, deterministic JSON."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(to_jsonable(value), sort_keys=True, ensure_ascii=False, indent=2)
    fd, tmp = tempfile.mkstemp(dir=str(target.parent), prefix=".nova-", suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write(payload)
            handle.write("\n")
        os.replace(tmp, target)
    except BaseException:
        if os.path.exists(tmp):
            os.unlink(tmp)
        raise
    return target


def read_json(path: str | os.PathLike[str]) -> Any:
    """Read a JSON document from ``path``."""
    return json.loads(Path(path).read_text(encoding="utf-8"))


def append_jsonl(path: str | os.PathLike[str], value: Any) -> Path:
    """Append one deterministic JSON line to ``path``."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("a", encoding="utf-8") as handle:
        handle.write(canonical_json(value))
        handle.write("\n")
    return target


def write_jsonl(path: str | os.PathLike[str], rows: Iterable[Any]) -> Path:
    """Write an iterable of records as JSONL, one deterministic line each."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(canonical_json(row))
            handle.write("\n")
    return target


def read_jsonl(path: str | os.PathLike[str]) -> list[Any]:
    """Read all records from a JSONL file (missing file -> empty list)."""
    target = Path(path)
    if not target.exists():
        return []
    rows: list[Any] = []
    for line in target.read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows
