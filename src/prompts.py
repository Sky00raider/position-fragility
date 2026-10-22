"""Prompt templates (doc §7) and the assistant-turn prefill splice (doc §6.6).

Every prompt is built here as text with the model's chat template, then tokenized
ONCE with add_special_tokens=False. Both backends receive these token ids, so the
pipeline is identical on vLLM and HF (and Llama does not get a double BOS).
"""
from __future__ import annotations

from .parse import THINK_OPEN, Hop

OPEN_BOOK = """Answer the question using only the paragraphs below. Reason one hop at a time.
Write each hop on its own line as "Hop N: <sub-question> -> <answer>".
After the last hop, write "Final answer: <short answer>".

Paragraphs:
{paragraphs}

Question: {question}"""

# Closed-book filter: same instructions, no paragraphs.
CLOSED_BOOK = """Answer the question. Reason one hop at a time.
Write each hop on its own line as "Hop N: <sub-question> -> <answer>".
After the last hop, write "Final answer: <short answer>".

Question: {question}"""


def format_paragraphs(paragraphs: list[dict]) -> str:
    return "\n\n".join(f"Title: {p['title']}\n{p['paragraph_text']}" for p in paragraphs)


def user_message(item: dict, closed_book: bool = False) -> str:
    if closed_book:
        return CLOSED_BOOK.format(question=item["question"])
    return OPEN_BOOK.format(paragraphs=format_paragraphs(item["context"]), question=item["question"])


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


def prefill_text(prefix: str, think: bool) -> str:
    """What goes after the assistant header. Think mode: prefix goes inside the think block."""
    return (THINK_OPEN + "\n" + prefix) if think else prefix


def encode(tok, text: str) -> list[int]:
    return tok(text, add_special_tokens=False)["input_ids"]


def build_ids(tok, model_id: str, user_msg: str, think: bool, prefill: str = "") -> list[int]:
    """Full prompt token ids: chat template + optional prefill (no EOS, no new user turn)."""
    return encode(tok, chat_prefix_text(tok, model_id, user_msg, think) + prefill)
