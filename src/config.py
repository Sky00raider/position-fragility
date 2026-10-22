"""Load a run config (one yaml per model x mode) and the HF token from .env.

The token is put into the environment for huggingface_hub to pick up.
It is never returned, printed or written to any output file.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

import yaml
from dotenv import load_dotenv

REPO_ROOT = Path(__file__).resolve().parent.parent


@dataclass
class Decoding:
    temperature: float = 0.7
    top_p: float = 0.8
    top_k: int = 20
    max_tokens: int = 1024


@dataclass
class DataCfg:
    splits: list[str] = field(default_factory=lambda: ["dev"])
    hops: list[int] = field(default_factory=lambda: [3, 4])
    limit_per_split_hop: int | None = None  # None = all items


@dataclass
class EligibilityCfg:
    f1_threshold: float = 0.8
    debug_relaxed: bool = False   # CPU debug only: keep items that have K hop lines
    hop_source: str = "answer"    # "answer" (after </think>) or "think"
    # CPU debug only: if the model's chain is unusable, build a stand-in chain from the gold
    # hops so inject/continue/score can be exercised. NEVER set in a real config.
    debug_gold_chain: bool = False


@dataclass
class Config:
    run_name: str
    backend: str                  # "vllm" | "hf"
    model_id: str
    mode: str                     # "nothink" | "think"
    tokenizer_id: str | None = None
    revision: str | None = None
    quant: str = "awq4"
    gpu: str = "none"
    decoding: Decoding = field(default_factory=Decoding)
    closed_book_max_tokens: int | None = None   # None = same as decoding.max_tokens
    max_model_len: int = 8192
    gpu_memory_utilization: float = 0.90
    batch_size: int = 64
    seed: int = 0
    device: str = "cpu"           # hf backend only: "cpu" or "cuda"
    data: DataCfg = field(default_factory=DataCfg)
    eligibility: EligibilityCfg = field(default_factory=EligibilityCfg)
    bootstrap_reps: int = 2000

    @property
    def think(self) -> bool:
        return self.mode == "think"

    @property
    def run_dir(self) -> Path:
        return REPO_ROOT / "runs" / self.run_name


def load_config(path: str | Path) -> Config:
    load_dotenv(REPO_ROOT / ".env")  # sets HF_TOKEN in os.environ if present
    raw = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    cfg = Config(
        **{k: v for k, v in raw.items() if k not in ("decoding", "data", "eligibility")},
        decoding=Decoding(**raw.get("decoding", {})),
        data=DataCfg(**raw.get("data", {})),
        eligibility=EligibilityCfg(**raw.get("eligibility", {})),
    )
    if cfg.mode not in ("nothink", "think"):
        raise ValueError(f"mode must be nothink|think, got {cfg.mode}")
    if cfg.backend not in ("vllm", "hf"):
        raise ValueError(f"backend must be vllm|hf, got {cfg.backend}")
    return cfg


def hf_token_present() -> bool:
    """For status messages only. Never print the token itself."""
    return bool(os.environ.get("HF_TOKEN"))
