"""Vet planted values with an LLM judge before they are used (added 2026-10-24, after the audit).

Why: rule 2 (type tag match) compares coarse spaCy tags, so a state could replace a city (both GPE)
and a TV network could replace a mistagged person (ORG). An audit of 100 planted values found only
~50% the right kind of thing and ~41% plausible. The judge first names the specific kind of the
correct answer and of the candidate, then answers: same kind? believable answer to the sub-question?

The cache stores what the judge said (both answers), not a decision. Which answers make a candidate
acceptable (RULES) is chosen by scripts/vet_gate.py on half of a labeled reference set, checked on the
other half, and written to a rule file next to the cache; vet and inject read it from there.

One judge (a fixed model, greedy decoding) vets candidates for every run, so all models and modes
draw from the same vetted values. Cache and rule file are per judge AND per prompt version
(runs/vet/<judge run_name>_p<PROMPT_VERSION>.*), so changing the prompt never reuses old verdicts.

History: prompt v1 (two yes/no lines) failed the gate on the GPU box on 2026-10-24: too few
acceptances and city/country confusions (Marco Polo born in "France" accepted for "Venice").
"""
from __future__ import annotations

import json
import re

from .normalize import normalize
from .store import read_jsonl

PROMPT_VERSION = 2

PROMPT = """A multi-step question was split into sub-questions. For one sub-question we know the correct answer. We want to test a WRONG answer that a careful reader could still believe.

Sub-question: {subq}
Correct answer: {gold}
Candidate wrong answer: {cand}

First name the specific kind of each answer, for example: city, country, US state, region, county, river, person, musician, politician, sports team, sports league, TV network, record label, company, university, year, date, number, language, nationality.
Then judge:
- same_kind: is the candidate the same specific kind of thing as the correct answer? A city is not a country or a state; a person is not an organization.
- believable: could a reasonable reader believe the candidate answers the sub-question? It is wrong, but must not be absurd for this sub-question.

Reply with exactly four lines and nothing else:
kind_of_correct: <kind>
kind_of_candidate: <kind>
same_kind: yes or no
believable: yes or no"""

_YESNO = re.compile(r"(same_kind|believable)\s*:\s*\**\s*(yes|no)", re.IGNORECASE)
_KIND = re.compile(r"kind_of_(correct|candidate)\s*:\s*\**\s*([^\n*]+)", re.IGNORECASE)

# which judge answers make a candidate acceptable; the gate picks one (see module docstring)
RULES = {
    "both": lambda v: bool(v.get("same_kind")) and bool(v.get("believable")),
    "believable": lambda v: bool(v.get("believable")),
    "same_kind": lambda v: bool(v.get("same_kind")),
}


def cache_path(repo_root, judge_run_name: str):
    """One cache per judge and prompt version, so verdicts are never reused across either."""
    return repo_root / "runs" / "vet" / f"{judge_run_name}_p{PROMPT_VERSION}.jsonl"


def rule_path(repo_root, judge_run_name: str):
    return repo_root / "runs" / "vet" / f"{judge_run_name}_p{PROMPT_VERSION}.rule.json"


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
    """{'same_kind': bool|None, 'believable': bool|None, 'kind_correct', 'kind_candidate'}.
    A missing answer stays None, which every rule treats as no."""
    out = {"same_kind": None, "believable": None, "kind_correct": None, "kind_candidate": None}
    for name, val in _YESNO.findall(text or ""):
        if out[name.lower()] is None:
            out[name.lower()] = val.lower() == "yes"
    for which, kind in _KIND.findall(text or ""):
        if out[f"kind_{which.lower()}"] is None:
            out[f"kind_{which.lower()}"] = kind.strip().lower()
    return out


def load_cache(path) -> dict[str, dict]:
    """vet_key -> the judge's answers ({'same_kind', 'believable', ...})."""
    return {r["key"]: r for r in read_jsonl(path)}


def read_rule(path) -> str | None:
    """The decision rule the gate chose and verified, or None if the gate has not passed."""
    try:
        rule = json.loads(path.read_text(encoding="utf-8"))["rule"]
    except (OSError, ValueError, KeyError):
        return None
    return rule if rule in RULES else None


def accepted(cache: dict[str, dict], rule: str) -> dict[str, bool]:
    """vet_key -> accepted under `rule`: the `vetted` mapping pick_value expects."""
    return {k: RULES[rule](v) for k, v in cache.items()}
