"""Splice: prefix ends with one newline, no EOS / end-of-turn at the splice, think prefill stays open."""
import pytest

from src.parse import parse_hops
from src.prompts import build_ids, build_prefix, chat_prefix_text, prefill_text

MODEL = "Qwen/Qwen3-0.6B"
OWN = ("Let me work through it.\n"
       "Hop 1: Who founded X? -> **Jane Doe**\n"
       "Hop 2: Where was Jane Doe born? -> London.\n"
       "Hop 3: Which country is London in? -> England\n"
       "Final answer: England")


@pytest.fixture(scope="module")
def tok():
    transformers = pytest.importorskip("transformers")
    try:
        return transformers.AutoTokenizer.from_pretrained(MODEL)
    except Exception as e:  # offline and not cached
        pytest.skip(f"tokenizer unavailable: {e}")


def test_prefix_cut_and_replace():
    hops = parse_hops(OWN)
    p = build_prefix(OWN, hops, 2, "Paris")
    assert p.endswith("Hop 2: Where was Jane Doe born? -> Paris.\n")
    assert not p.endswith("\n\n") and "Hop 3" not in p
    assert p.startswith("Let me work through it.")   # model's own text kept verbatim
    clean = build_prefix(OWN, hops, 1)
    assert clean.endswith("-> **Jane Doe**\n")


@pytest.mark.parametrize("think", [False, True])
def test_no_eos_at_splice(tok, think):
    prefix = build_prefix(OWN, parse_hops(OWN), 2, "Paris")
    ids = build_ids(tok, MODEL, "Question?", think, prefill_text(prefix, think))
    template_len = len(build_ids(tok, MODEL, "Question?", think))
    tail = ids[template_len:]
    stop_ids = {tok.eos_token_id, tok.convert_tokens_to_ids("<|im_end|>"), tok.convert_tokens_to_ids("<|endoftext|>")}
    assert not stop_ids & set(tail)
    assert ids[:template_len] == build_ids(tok, MODEL, "Question?", think)   # splice did not re-tokenize the template
    decoded = tok.decode(ids)
    assert decoded.endswith(prefix)
    assert decoded.count("<|im_start|>user") == 1          # no new user turn
    if think:
        assert decoded.endswith("<|im_start|>assistant\n<think>\n" + prefix)
        assert "</think>" not in decoded
    else:
        assert decoded.endswith("</think>\n\n" + prefix)   # after the template's empty think block


def test_template_kwargs(tok):
    assert chat_prefix_text(tok, MODEL, "Q", False).endswith("<think>\n\n</think>\n\n")
    assert chat_prefix_text(tok, MODEL, "Q", True).endswith("<|im_start|>assistant\n")
