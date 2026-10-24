"""Planted-value vetting (src/vet.py) and its use in pick_value, run_judge and reuse_cells."""
from src import pipeline
from src.backends import Backend, Gen
from src.config import Config
from src.inject import pick_value
from src.store import read_jsonl, write_jsonl
import importlib.util
import json
from pathlib import Path

from src.vet import RULES, accepted, load_cache, parse_verdict, read_rule, resolve_refs, vet_key

both = RULES["both"]


def test_parse_verdict():
    v = parse_verdict("kind_of_correct: city\nkind_of_candidate: country\nsame_kind: no\nbelievable: yes")
    assert v["kind_correct"] == "city" and v["kind_candidate"] == "country"
    assert v["same_kind"] is False and v["believable"] is True
    assert both(parse_verdict("same_kind: yes\nbelievable: yes"))
    assert both(parse_verdict("Same_kind: **yes**\nBelievable: Yes"))
    assert not both(parse_verdict("same_kind: yes\nbelievable: no"))
    assert RULES["believable"](parse_verdict("same_kind: no\nbelievable: yes"))
    v = parse_verdict("I think so.")                      # unparseable counts as rejected by every rule
    assert v["same_kind"] is None and not any(rule(v) for rule in RULES.values())
    assert not both(parse_verdict("same_kind: yes"))      # half an answer is not an accept


def test_accepted_and_read_rule(tmp_path):
    cache = {"a": {"same_kind": True, "believable": False}, "b": {"same_kind": True, "believable": True}}
    assert accepted(cache, "both") == {"a": False, "b": True}
    assert accepted(cache, "same_kind") == {"a": True, "b": True}
    p = tmp_path / "r.json"
    assert read_rule(p) is None                           # no file: gate has not passed
    p.write_text(json.dumps({"rule": "believable"}))
    assert read_rule(p) == "believable"
    p.write_text(json.dumps({"rule": "made_up"}))
    assert read_rule(p) is None


def load_gate():
    spec = importlib.util.spec_from_file_location("vet_gate", Path(__file__).parents[1] / "scripts" / "vet_gate.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_gate_choose_rule():
    gate = load_gate()
    # 'both' is precise but accepts little; 'believable' accepts more but is sloppy
    rows = []
    for i in range(20):
        ok = i < 8
        rows.append({"ok": ok, "judge": {"both": ok and i < 4, "believable": ok or i < 14, "same_kind": i % 2 == 0}})
    assert gate.choose_rule(rows, min_accept=0.15) == "both"      # 4/20 = 20% accepted, precision 1.0
    assert gate.choose_rule(rows, min_accept=0.5) == "believable"  # only rule over 50%
    assert gate.choose_rule(rows, min_accept=0.9) is None


def test_resolve_refs():
    assert resolve_refs("Who founded #1 in #2?", ["Acme", "1990"]) == "Who founded Acme in 1990?"
    assert resolve_refs("no refs", ["a"]) == "no refs"


def make_item():
    return {"id": "a", "question": "q?", "answer": "z", "answer_aliases": [], "K": 3,
            "hops": [{"question": "What city is #2 in?", "answer": "Lyon", "tag": "GPE"},
                     {"question": "x", "answer": "Rhone", "tag": "LOC"},
                     {"question": "y", "answer": "z", "tag": "OTHER"}],
            "context": [{"title": "T", "paragraph_text": "filler"}],
            "distractor_candidates": [{"text": "Texas", "tag": "GPE", "source": "title"},
                                      {"text": "Nice", "tag": "GPE", "source": "title"},
                                      {"text": "Brest", "tag": "GPE", "source": "title"}]}


def test_pick_value_takes_first_accepted_candidate():
    it = make_item()
    first, _, _ = pick_value(it, 1, "Lyon", "in_context", "", [])
    order = [first] + [c for c in ("Texas", "Nice", "Brest") if c != first]
    subq = "What city is Rhone in?"
    vetted = {vet_key(subq, "Lyon", order[0]): False, vet_key(subq, "Lyon", order[1]): True}
    v, reason, n = pick_value(it, 1, "Lyon", "in_context", "", [], vetted=vetted)
    assert v == order[1] and reason is None and n == 2


def test_pick_value_unvetted_and_all_rejected():
    it = make_item()
    v, reason, _ = pick_value(it, 1, "Lyon", "in_context", "", [], vetted={})
    assert v is None and reason == "not_vetted"
    subq = "What city is Rhone in?"
    all_no = {vet_key(subq, "Lyon", c): False for c in ("Texas", "Nice", "Brest")}
    v, reason, n = pick_value(it, 1, "Lyon", "in_context", "", [], vetted=all_no)
    assert v is None and reason == "vet_rejected_all" and n == 3


class JudgeBackend(Backend):
    """Accepts any candidate except 'Texas'."""
    def __init__(self):
        self.n = 0

    def generate(self, prompt_ids, seeds, max_tokens):
        self.n += len(prompt_ids)
        return [Gen("same_kind: yes\nbelievable: yes", 8, "stop") for _ in prompt_ids]


def test_run_judge_caches_and_skips_known(tmp_path, monkeypatch):
    judge = Config(run_name="j", backend="hf", model_id="fake", mode="nothink", batch_size=2)
    fake = JudgeBackend()
    monkeypatch.setattr(pipeline, "make_backend", lambda cfg: fake)
    monkeypatch.setattr(pipeline, "load_tokenizer", lambda cfg: None)
    monkeypatch.setattr(pipeline, "build_ids", lambda *a, **k: [1, 2])
    cache_path = tmp_path / "vet_cache.jsonl"
    triples = [("q", "g", "a"), ("q", "g", "b"), ("q", "g", "a"), ("q", "g", "c")]
    pipeline.run_judge(judge, triples, cache_path)
    assert fake.n == 3                                   # duplicate judged once
    pipeline.run_judge(judge, triples, cache_path)
    assert fake.n == 3                                   # all cached now
    assert both(load_cache(cache_path)[vet_key("q", "g", "b")])


def test_reuse_cells_only_identical_prompts(tmp_path, monkeypatch):
    cfg = Config(run_name="new", backend="hf", model_id="fake", mode="nothink", reuse_cells_from="old")
    cfg_dir = tmp_path / "runs"
    (cfg_dir / "old").mkdir(parents=True)
    old_rows = [
        {"item_id": "a", "condition": "noprefix", "k": 0, "error_type": None, "prefix": "", "prompt_len": 3, "text": "n"},
        {"item_id": "a", "condition": "clean", "k": 1, "error_type": None, "prefix": "Hop 1: x\n", "prompt_len": 5, "text": "c"},
        {"item_id": "a", "condition": "clean", "k": 2, "error_type": None, "prefix": "OLD\n", "prompt_len": 5, "text": "c2"},
        {"item_id": "a", "condition": "inject", "k": 1, "error_type": "in_context", "prefix": "Hop 1: y\n", "prompt_len": 5, "text": "i"},
    ]
    write_jsonl(cfg_dir / "old" / "continuations.jsonl", old_rows)
    cells = [
        {"item_id": "a", "condition": "noprefix", "k": 0, "error_type": None, "prefix": "", "prompt_ids": [1, 2, 3]},
        {"item_id": "a", "condition": "clean", "k": 1, "error_type": None, "prefix": "Hop 1: x\n", "prompt_ids": [1] * 5},
        {"item_id": "a", "condition": "clean", "k": 2, "error_type": None, "prefix": "NEW\n", "prompt_ids": [1] * 5},
        {"item_id": "a", "condition": "inject", "k": 1, "error_type": "in_context", "prefix": "Hop 1: y\n", "prompt_ids": [1] * 5},
    ]
    out = cfg_dir / "new" / "continuations.jsonl"
    monkeypatch.setattr(Config, "run_dir", property(lambda self: cfg_dir / self.run_name))   # run folder in tmp
    pipeline.reuse_cells(cfg, cells, out)
    got = {(r["condition"], r["k"]) for r in read_jsonl(out)}
    assert got == {("noprefix", 0), ("clean", 1)}       # changed prefix and inject cells are regenerated
