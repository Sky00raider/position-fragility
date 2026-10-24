"""Vet planted values with an LLM judge before they are used (added 2026-10-24, after the audit).

Why: rule 2 (type tag match) compares coarse spaCy tags, so a state could replace a city (both GPE)
and a TV network could replace a mistagged person (ORG). An audit of 100 planted values found only
~50% the right kind of thing and ~41% plausible. The judge asks two questions per candidate:
same kind of thing as the gold hop answer? believable answer to the sub-question?
A candidate is used only if the judge says yes to both.

One judge (a fixed model, greedy decoding) vets candidates for every run, so all models and modes
draw from the same vetted values. Verdicts are cached in runs/vet/<judge run_name>.jsonl, keyed by
(sub-question, gold hop answer, candidate), and shared across runs.
"""
from __future__ import annotations

import re

from .normalize import normalize
from .store import read_jsonl


def cache_path(repo_root, judge_run_name: str):
    """One cache per judge, so verdicts of one judge are never reused for another."""
    return repo_root / "runs" / "vet" / f"{judge_run_name}.jsonl"


PROMPT = """A multi-step question was split into sub-questions. For one sub-question we know the correct answer, and we want to test a WRONG answer that a careful reader could still believe.

Sub-question: {subq}
Correct answer: {gold}
Candidate wrong answer: {cand}

1. Is the candidate the same kind of thing as the correct answer? Be specific: a city is not a state or country, a person is not an organization, a TV network is not a record label, a year is not a decade.
2. Could a reasonable reader believe the candidate answers the sub-question? Answer no if it is absurd for this sub-question (for example a pop singer as an 18th-century diplomat, or a painting as a person).

Reply with exactly two lines and nothing else:
same_kind: yes or no
believable: yes or no"""

_LINE = re.compile(r"(same_kind|believable)\s*:\s*\**\s*(yes|no)", re.IGNORECASE)


def resolve_refs(subq: str, gold_hops: list[str]) -> str:
    """MuSiQue sub-questions refer to earlier hops as #1, #2: fill in the gold answers."""
    def sub(m):
        i = int(m.group(1)) - 1
        return gold_hops[i] if 0 <= i < len(gold_hops) else m.group(0)
    return re.sub(r"#(\d+)", sub, subq)


def vet_key(subq: str, gold: str, cand: str) -> str:
    return " || ".join(normalize(x) for x in (subq, gold, cand))


def judge_message(subq: str, gold: str, cand: str) -> str:
    return PROMPT.format(subq=subq.strip(), gold=gold.strip(), cand=cand.strip())


def parse_verdict(text: str) -> dict:
    """{'same_kind': bool|None, 'believable': bool|None, 'ok': bool}. Unparseable counts as not ok."""
    out = {"same_kind": None, "believable": None}
    for name, val in _LINE.findall(text or ""):
        out[name.lower()] = out[name.lower()] if out[name.lower()] is not None else val.lower() == "yes"
    out["ok"] = bool(out["same_kind"]) and bool(out["believable"])
    return out


def load_cache(path) -> dict[str, bool]:
    """vet_key -> ok (the judge accepted the candidate)."""
    return {r["key"]: r["ok"] for r in read_jsonl(path)}
