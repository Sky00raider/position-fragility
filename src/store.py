"""Append-only jsonl files, cell keys and per-cell seeds (resumability lives here)."""
from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path
from typing import Iterable, Iterator


def read_jsonl(path: str | Path) -> list[dict]:
    path = Path(path)
    if not path.exists():
        return []
    rows = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError:
                    # a run killed mid-write can leave a partial last line; skip it
                    continue
    return rows


def iter_jsonl(path: str | Path) -> Iterator[dict]:
    yield from read_jsonl(path)


def append_jsonl(path: str | Path, rows: Iterable[dict]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
        f.flush()


def write_jsonl(path: str | Path, rows: Iterable[dict]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def cell_key(row: dict) -> str:
    """Unique id of one cell within a run: item x condition x hop x error type."""
    return "|".join(str(row.get(f)) for f in ("item_id", "condition", "k", "error_type"))


def done_keys(path: str | Path) -> set[str]:
    return {cell_key(r) for r in read_jsonl(path)}


def cell_seed(*parts) -> int:
    """Stable across processes and machines (built-in hash() is not)."""
    s = "|".join(str(p) for p in parts)
    return int(hashlib.sha256(s.encode("utf-8")).hexdigest()[:8], 16)


def git_hash() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], stderr=subprocess.DEVNULL,
            cwd=Path(__file__).resolve().parent,
        ).decode().strip()
    except Exception:
        return "nogit"
