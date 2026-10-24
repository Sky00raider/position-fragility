"""Splice: prefix ends with one newline, no EOS / end-of-turn at the splice, think prefill stays open,
exactly one BOS (Llama), template tokens unchanged by the splice. Run for every real tokenizer."""
import pytest

from src.parse import parse_hops
from src.prompts import build_ids, build_prefix, chat_prefix_text, is_qwen3, prefill_text

QWEN_SMALL = "Qwen/Qwen3-0.6B"
QWEN_8B = "Qwen/Qwen3-8B-AWQ"
LLAMA = "hugging-quants/Meta-Llama-3.1-8B-Instruct-AWQ-INT4"
OWN = ("Let me work through it.\n"
       "Hop 1: Who founded X? -> **Jane Doe**\n"
       "Hop 2: Where was Jane Doe born? -> London.\n"
       "Hop 3: Which country is London in? -> England\n"
       "Final answer: England")
END_TOKENS = ["<|im_end|>", "<|endoftext|>", "<|eot_id|>", "<|end_of_text|>", "<|eom_id|>"]

_cache = {}


def get_tok(model):
    transformers = pytest.importorskip("transformers")
    if model not in _cache:
        try:
            _cache[model] = transformers.AutoTokenizer.from_pretrained(model)
        except Exception as e:  # offline and not cached
            pytest.skip(f"tokenizer unavailable: {e}")
    return _cache[model]


CASES = [(QWEN_SMALL, False), (QWEN_SMALL, True), (QWEN_8B, False), (QWEN_8B, True), (LLAMA, False)]


def test_prefix_cut_and_replace():
    hops = parse_hops(OWN)
    p = build_prefix(OWN, hops, 2, "Paris")
    assert p.endswith("Hop 2: Where was Jane Doe born? -> Paris.\n")
    assert not p.endswith("\n\n") and "Hop 3" not in p
    assert p.startswith("Let me work through it.")   # model's own text kept verbatim
    clean = build_prefix(OWN, hops, 1)
    assert clean.endswith("-> **Jane Doe**\n")


@pytest.mark.parametrize("model,think", CASES)
def test_no_eos_at_splice(model, think):
    tok = get_tok(model)
    prefix = build_prefix(OWN, parse_hops(OWN), 2, "Paris")
    ids = build_ids(tok, model, "Question?", think, prefill_text(prefix, think))
    template = build_ids(tok, model, "Question?", think)
    tail = ids[len(template):]
    stop_ids = {tok.eos_token_id} | {tok.convert_tokens_to_ids(t) for t in END_TOKENS if t in tok.get_vocab()}
    assert not stop_ids & set(tail), "end-of-turn token inside the prefill"
    assert ids[:len(template)] == template, "splice changed the template tokens"
    decoded = tok.decode(ids)
    assert decoded.endswith(prefix)
    if tok.bos_token_id is not None:
        assert ids.count(tok.bos_token_id) <= 1, "double BOS"
    if think:
        assert decoded.endswith("<|im_start|>assistant\n<think>\n" + prefix)
        assert "</think>" not in decoded
    elif is_qwen3(model):
        assert decoded.endswith("</think>\n\n" + prefix)   # after the template's empty think block
        assert decoded.count("<|im_start|>user") == 1      # no new user turn
    else:
        assert decoded.endswith("<|start_header_id|>assistant<|end_header_id|>\n\n" + prefix)
        assert decoded.count("<|start_header_id|>user") == 1


@pytest.mark.parametrize("model", [QWEN_SMALL, QWEN_8B])
def test_qwen_template_kwargs(model):
    tok = get_tok(model)
    assert chat_prefix_text(tok, model, "Q", False).endswith("<think>\n\n</think>\n\n")
    assert chat_prefix_text(tok, model, "Q", True).endswith("<|im_start|>assistant\n")


def test_trace_mention_and_prefix():
    from src.prompts import build_trace_prefix, find_trace_mention
    trace = ("Okay, the question asks about the performer. The paragraph says the album is by Charles Mingus. "
             "So the performer is Charles Mingus.\nNext, Mingus was born in Nogales, Arizona. The capital is Phoenix.")
    s, e, m = find_trace_mention(trace, ["charles mingus", "Mingus"])
    assert m == "Charles Mingus" and trace[s:e] == m                      # earliest whole-word mention
    clean = build_trace_prefix(trace, s, e)
    assert clean.endswith("the album is by Charles Mingus.")              # cut at the end of that sentence
    bad = build_trace_prefix(trace, s, e, "Miles Davis")
    assert bad.endswith("the album is by Miles Davis.") and "Mingus" not in bad
    assert find_trace_mention(trace, ["Arizon"]) is None                   # whole words only
    s2, e2, _ = find_trace_mention(trace, ["Phoenix"])
    assert build_trace_prefix(trace, s2, e2) == trace                       # sentence ending at end of text
    assert build_trace_prefix("no end here Phoenix", 12, 19) is None       # unfinished sentence -> drop
    assert find_trace_mention("they chose the Soviets.", ["the Soviets"])[2] == "the Soviets"
    assert find_trace_mention("they chose Soviets.", ["the Soviets"])[2] == "Soviets"   # leading "the" optional


@pytest.mark.parametrize("model", [QWEN_8B])
def test_trace_splice_tokens(model):
    tok = get_tok(model)
    prefix = "Okay, let me think. The album is by Miles Davis."
    ids = build_ids(tok, model, "Question?", True, prefill_text(prefix, True))
    decoded = tok.decode(ids)
    assert decoded.endswith("<|im_start|>assistant\n<think>\n" + prefix) and "</think>" not in decoded


def test_trace_requires_think(tmp_path):
    from src.config import load_config
    p = tmp_path / "c.yaml"
    p.write_text("run_name: x\nbackend: hf\nmodel_id: m\nmode: nothink\ninjection_site: trace\n")
    with pytest.raises(ValueError):
        load_config(p)
    p.write_text("run_name: x\nbackend: hf\nmodel_id: m\nmode: think\ninjection_site: trace\n")
    assert load_config(p).mode_label == "think_trace"
