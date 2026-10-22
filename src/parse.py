"""Parse model output: think/answer split, hop lines, final answer.

Output format we ask for (doc §7):
    Hop 1: <sub-question> -> <answer>
    ...
    Final answer: <short answer>
"""
from __future__ import annotations

import re
from dataclasses import dataclass

THINK_OPEN, THINK_CLOSE = "<think>", "</think>"

# "Hop 2: who ... -> Paris", tolerant to bold markers and "Hop 2." / "Hop 2 -"
HOP_RE = re.compile(r"^[ \t>*_-]*\**\s*hop\s*(\d+)\s*\**\s*[:.\-)]\s*\**(.*)$", re.IGNORECASE)
ARROW_RE = re.compile(r"\s*(?:->|→|=>)\s*")
FINAL_RE = re.compile(r"\**\s*final\s+answer\s*\**\s*[:：]\s*\**", re.IGNORECASE)


@dataclass
class Hop:
    n: int            # hop number as written by the model
    question: str
    answer: str
    line_start: int   # char offsets in the text that was parsed
    line_end: int     # end of line, excluding the newline
    ans_start: int    # span of the answer inside the text
    ans_end: int


def split_think(text: str, think_mode: bool) -> tuple[str, str, bool]:
    """Return (think_part, answer_part, think_closed).

    think mode: everything before </think> is thinking; the answer part is what follows.
    If </think> is missing (truncated), the answer part is empty.
    nothink mode: whole text is the answer (a stray </think> is still honored).
    """
    if THINK_CLOSE in text:
        think, answer = text.split(THINK_CLOSE, 1)
        think = think.replace(THINK_OPEN, "", 1)
        return think, answer, True
    if think_mode:
        return text.replace(THINK_OPEN, "", 1), "", False
    return "", text, True


def _clean_answer(s: str) -> str:
    s = s.strip()
    s = re.sub(r"</?[A-Za-z_ -]{1,20}>", "", s).strip()   # stray <answer>-style tags
    s = re.sub(r"^\\boxed\{(.*)\}$", r"\1", s)
    s = s.strip("*_` ")
    s = re.sub(r"^(?:the\s+)?answer\s+is\s*:?\s*", "", s, flags=re.IGNORECASE)
    s = s.strip().strip("\"'“”‘’`*_ ")
    s = s.rstrip(".;,")
    return s.strip().strip("\"'“”‘’ ")


def parse_hops(text: str) -> list[Hop]:
    """All hop lines in order. The answer is the text after the LAST arrow on the line."""
    hops, pos = [], 0
    for line in text.split("\n"):
        start, end = pos, pos + len(line)
        pos = end + 1
        m = HOP_RE.match(line)
        if not m:
            continue
        body_off = start + m.start(2)
        body = m.group(2)
        arrows = list(ARROW_RE.finditer(body))
        if not arrows:
            continue
        last = arrows[-1]
        q = body[: last.start()].strip().strip("*")
        raw_ans = body[last.end():]
        # answer span: trim whitespace and markdown/punctuation that surround it
        lead = len(raw_ans) - len(raw_ans.lstrip(" *_`\"'"))
        core = raw_ans.strip(" *_`\"'").rstrip(".")
        core = core.strip(" *_`\"'")
        a_start = body_off + last.end() + lead
        hops.append(Hop(
            n=int(m.group(1)), question=q, answer=_clean_answer(core),
            line_start=start, line_end=end,
            ans_start=a_start, ans_end=a_start + len(core),
        ))
    return hops


def hops_well_formed(hops: list[Hop], K: int) -> bool:
    """Exactly K hop lines numbered 1..K with non-empty answers."""
    return [h.n for h in hops] == list(range(1, K + 1)) and all(h.answer for h in hops)


def extract_final(answer_part: str) -> str | None:
    """Last 'Final answer:' in the answer part. None if absent or empty."""
    matches = list(FINAL_RE.finditer(answer_part))
    if not matches:
        return None
    rest = answer_part[matches[-1].end():]
    lines = [ln for ln in rest.split("\n") if ln.strip()]
    if not lines:
        return None
    ans = _clean_answer(lines[0])
    return ans or None
