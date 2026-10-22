"""continue is resumable: a crash mid-run loses nothing, a rerun only generates missing cells."""
import pytest

from src import pipeline
from src.backends import Backend, Gen
from src.config import Config
from src.store import cell_key, read_jsonl


class FakeBackend(Backend):
    def __init__(self, crash_after_calls=None):
        self.calls, self.crash_after = [], crash_after_calls

    def generate(self, prompt_ids, seeds, max_tokens):
        if self.crash_after is not None and len(self.calls) >= self.crash_after:
            raise RuntimeError("simulated crash")
        self.calls.append(len(prompt_ids))
        return [Gen(f"Final answer: x{s}", 3, "stop") for s in seeds]


def make_cells(n):
    return [{"item_id": f"it{i // 3}", "condition": "inject", "k": i % 3 + 1, "error_type": "in_context",
             "prompt_ids": [1, 2, 3], "max_tokens": 8} for i in range(n)]


def test_resume_after_crash(tmp_path, monkeypatch):
    cfg = Config(run_name="test", backend="hf", model_id="fake", mode="nothink", batch_size=4)
    out = tmp_path / "continuations.jsonl"
    cells = make_cells(10)

    crashing = FakeBackend(crash_after_calls=1)        # finishes batch 1, crashes on batch 2
    monkeypatch.setattr(pipeline, "make_backend", lambda cfg: crashing)
    with pytest.raises(RuntimeError):
        pipeline.generate_cells(cfg, cells, out, "continue")
    assert len(read_jsonl(out)) == 4

    fresh = FakeBackend()
    monkeypatch.setattr(pipeline, "make_backend", lambda cfg: fresh)
    pipeline.generate_cells(cfg, cells, out, "continue")
    rows = read_jsonl(out)
    keys = [cell_key(r) for r in rows]
    assert len(keys) == len(set(keys)) == 10            # no duplicates, everything done
    assert sum(fresh.calls) == 6                        # only the missing cells were generated

    again = FakeBackend()
    monkeypatch.setattr(pipeline, "make_backend", lambda cfg: again)
    pipeline.generate_cells(cfg, cells, out, "continue")
    assert again.calls == [] and len(read_jsonl(out)) == 10   # nothing left to do


def test_partial_last_line_is_ignored(tmp_path):
    p = tmp_path / "x.jsonl"
    p.write_text('{"item_id": "a", "condition": "clean", "k": 1, "error_type": null}\n{"item_id": "b", "cond',
                 encoding="utf-8")
    assert len(read_jsonl(p)) == 1


def test_same_seed_on_rerun(tmp_path, monkeypatch):
    cfg = Config(run_name="test", backend="hf", model_id="fake", mode="nothink", batch_size=2)
    fake = FakeBackend()
    monkeypatch.setattr(pipeline, "make_backend", lambda cfg: fake)
    pipeline.generate_cells(cfg, make_cells(2), tmp_path / "a.jsonl", "continue")
    pipeline.generate_cells(cfg, make_cells(2), tmp_path / "b.jsonl", "continue")
    assert [r["seed"] for r in read_jsonl(tmp_path / "a.jsonl")] == [r["seed"] for r in read_jsonl(tmp_path / "b.jsonl")]
