"""Prompt templates (doc §7) and the assistant-turn prefill splice (doc §6.6).

Every prompt is built here as text with the model's chat template, then tokenized
ONCE with add_special_tokens=False. Both backends receive these token ids, so the
pipeline is identical on vLLM and HF (and Llama does not get a double BOS).
"""
from __future__ import annotations

import re

from .parse import THINK_OPEN, Hop

# v2 (after the GPU pilot, 2026-10-23): v1 let Qwen3-8B split questions into 5-11 hops and answer
# hops in full sentences, so almost no own-chain matched the gold decomposition. v2 states the hop
# count, asks for short answers and shows one worked example (made up, not from MuSiQue).
INSTRUCTIONS = """Reason one hop at a time. This question needs exactly {K} hops.
Write each hop on its own line as "Hop N: <sub-question> -> <answer>", where <answer> is a short name, date or number, not a sentence.
After hop {K}, write "Final answer: <short answer>".

Example of the format (a 3-hop question):
Question: What is the official language of the country whose capital is the city where the Eiffel Tower stands?
Hop 1: In which city does the Eiffel Tower stand? -> Paris
Hop 2: Which country has Paris as its capital? -> France
Hop 3: What is the official language of France? -> French
Final answer: French"""

OPEN_BOOK = """Answer the question using only the paragraphs below. """ + INSTRUCTIONS + """

Paragraphs:
{paragraphs}

Question: {question}"""

# Closed-book filter: same instructions, no paragraphs.
CLOSED_BOOK = """Answer the question. """ + INSTRUCTIONS + """

Question: {question}"""


def format_paragraphs(paragraphs: list[dict]) -> str:
    return "\n\n".join(f"Title: {p['title']}\n{p['paragraph_text']}" for p in paragraphs)


def user_message(item: dict, closed_book: bool = False) -> str:
    if closed_book:
        return CLOSED_BOOK.format(question=item["question"], K=item["K"])
    return OPEN_BOOK.format(paragraphs=format_paragraphs(item["context"]), question=item["question"], K=item["K"])


def is_qwen3(model_id: str) -> bool:
    return "qwen3" in model_id.lower()


def chat_prefix_text(tok, model_id: str, user_msg: str, think: bool) -> str:
    """Chat-templated text up to and including the assistant header.

    Qwen3: enable_thinking=False makes the template end with an empty
    "<think>\\n\\n</think>\\n\\n" block; enable_thinking=True ends at the assistant header
    and the model writes <think> itself.
    """
    kw = {"enable_thinking": think} if is_qwen3(model_id) else {}
    messages = [{"role": "user", "content": user_msg}]
    return tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, **kw)


def build_prefix(answer_text: str, hops: list[Hop], k: int, new_value: str | None = None) -> str:
    """Model's own answer text cut after hop line k; hop-k answer optionally replaced.

    `hops` must be parsed from exactly `answer_text`. Always ends with a single "\\n".
    Only the answer span on line k is changed; everything else is the model's own text.
    """
    h = hops[k - 1]
    text = answer_text[: h.line_end]
    if new_value is not None:
        text = text[: h.ans_start] + new_value + text[h.ans_end:]
    return text.lstrip("\n ").rstrip() + "\n"


# ---- trace injection (thinking mode, design doc §6.6 alternative; von Recum et al.) ----
# The "hops" site prefills the model's tidy hop summary inside <think>; Qwen3 then treats its
# reasoning as finished (median 110 new tokens, no deliberation). The "trace" site corrupts the
# model's OWN thinking text instead: first mention of the hop-k answer, cut at the end of that
# sentence, so the model resumes mid-thought and keeps reasoning.

SENTENCE_END = re.compile(r"[.!?](?=\s|$)|\n")


def _variants(s: str) -> list[str]:
    s = re.sub(r"\([^)]*\)", " ", s).strip(" .,;:\"'*")
    out = [s]
    if s.lower().startswith("the "):
        out.append(s[4:])
    return [v for v in out if len(v) >= 2]


def find_trace_mention(think_text: str, candidates: list[str]) -> tuple[int, int, str] | None:
    """Earliest whole-word, case-insensitive mention of any candidate string in the thinking text.
    Returns (start, end, matched text) or None."""
    best = None
    for c in candidates:
        for v in _variants(c):
            m = re.search(r"(?<!\w)" + re.escape(v) + r"(?!\w)", think_text, re.IGNORECASE)
            if m and (best is None or m.start() < best[0]):
                best = (m.start(), m.end(), m.group(0))
    return best


def build_trace_prefix(think_text: str, start: int, end: int, new_value: str | None = None) -> str | None:
    """Thinking text up to the end of the sentence containing [start, end); the mention optionally
    replaced. None if no sentence end follows (the mention is in the last, unfinished sentence)."""
    m = SENTENCE_END.search(think_text, end)
    if m is None:
        return None
    text = think_text[: m.end()]
    if new_value is not None:
        text = text[:start] + new_value + text[end:]
    return text


def prefill_text(prefix: str, think: bool) -> str:
    """What goes after the assistant header. Think mode: prefix goes inside the think block."""
    return (THINK_OPEN + "\n" + prefix) if think else prefix


def encode(tok, text: str) -> list[int]:
    return tok(text, add_special_tokens=False)["input_ids"]


def build_ids(tok, model_id: str, user_msg: str, think: bool, prefill: str = "") -> list[int]:
    """Full prompt token ids: chat template + optional prefill (no EOS, no new user turn)."""
    return encode(tok, chat_prefix_text(tok, model_id, user_msg, think) + prefill)
