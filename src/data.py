"""MuSiQue-Ans download, 3/4-hop filtering, item contexts and answer type tags.

Each prepared item (data/items/musique_{split}_34hop.jsonl) holds:
  id, split, composition (e.g. "3hop1"), K, question, answer, answer_aliases,
  hops: [{question, answer, tag, depends_on, remaining_dep}],
  is_linear, context: [{title, paragraph_text, is_supporting}] (gold + 4 distractors, shuffled),
  distractor_candidates: [{text, tag, source}]   # used by in_context injection
"""
from __future__ import annotations

import hashlib
import random
import re
import zipfile
from collections import Counter
from pathlib import Path

from tqdm import tqdm

from .config import REPO_ROOT
from .store import read_jsonl, write_jsonl

RAW_DIR = REPO_ROOT / "data" / "raw" / "musique"
ITEMS_DIR = REPO_ROOT / "data" / "items"
GDRIVE_ID = "1tGdADlNjWFaHLeZZGShh2IRcpO6Lv24h"   # official StonyBrookNLP/musique download_data.sh
HF_MIRROR = "dgslibisey/MuSiQue"                    # fallback: same musique_ans_v1.0_*.jsonl files
SEED = 13
N_DISTRACTORS = 4

# ---------------------------------------------------------------- download

def raw_path(split: str) -> Path:
    return RAW_DIR / f"musique_ans_v1.0_{split}.jsonl"


def _download_gdrive() -> None:
    import gdown
    zip_path = RAW_DIR / "musique_v1.0.zip"
    if not zip_path.exists():
        if gdown.download(id=GDRIVE_ID, output=str(zip_path), quiet=False) is None:
            raise RuntimeError("gdown failed")
    with zipfile.ZipFile(zip_path) as z:
        for name in z.namelist():
            base = Path(name).name
            if base.startswith("musique_ans_v1.0_") and base.endswith(".jsonl"):
                (RAW_DIR / base).write_bytes(z.read(name))


def _download_hf() -> None:
    from huggingface_hub import hf_hub_download
    for split in ("train", "dev"):
        hf_hub_download(HF_MIRROR, raw_path(split).name, repo_type="dataset", local_dir=RAW_DIR)


def download() -> None:
    """Official Google Drive zip first; HF mirror with the same files if Drive refuses (quota)."""
    if raw_path("train").exists() and raw_path("dev").exists():
        print("MuSiQue already downloaded")
        return
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    try:
        _download_gdrive()
    except Exception as e:
        print(f"Google Drive download failed ({type(e).__name__}); using HF mirror {HF_MIRROR}")
        _download_hf()
    for split in ("train", "dev"):
        if not raw_path(split).exists():
            raise FileNotFoundError(f"{raw_path(split)} missing after unzip")


# ---------------------------------------------------------------- answer types

MONTHS = "january|february|march|april|may|june|july|august|september|october|november|december"
DATE_PATTERNS = [
    ("DATE_MDY", rf"^(?:{MONTHS})\s+\d{{1,2}},?\s+\d{{3,4}}$"),
    ("DATE_DMY", rf"^\d{{1,2}}\s+(?:{MONTHS}),?\s+\d{{3,4}}$"),
    ("DATE_MY", rf"^(?:{MONTHS}),?\s+\d{{3,4}}$"),
    ("DATE_MD", rf"^(?:{MONTHS})\s+\d{{1,2}}$"),
    ("DATE_ISO", r"^\d{4}-\d{2}-\d{2}$"),
]
YEAR_RE = re.compile(r"^(1[0-9]{3}|20[0-9]{2})$")
NUMBER_RE = re.compile(r"^-?\d[\d,]*(\.\d+)?$")
YEAR_IN_TEXT = re.compile(r"\b(1[0-9]{3}|20[0-9]{2})\b")


def surface_tag(text: str) -> str | None:
    """Regex type for years, dates and plain numbers; None if it is not one of those."""
    t = text.strip()
    if YEAR_RE.match(t):
        return "YEAR"
    for tag, pat in DATE_PATTERNS:
        if re.match(pat, t, re.IGNORECASE):
            return tag
    if NUMBER_RE.match(t):
        return "NUMBER"
    return None


def load_nlp():
    import spacy
    return spacy.load("en_core_web_sm", disable=["parser", "lemmatizer", "tagger", "attribute_ruler"])


def ner_tag(text: str, doc=None, nlp=None) -> str:
    """NER label of `text`: the entity overlapping its first mention in `doc`, else NER on text alone."""
    if doc is not None:
        i = doc.text.lower().find(text.lower())
        if i >= 0:
            j = i + len(text)
            for ent in doc.ents:
                if ent.start_char < j and ent.end_char > i:
                    return ent.label_
    if nlp is not None:
        ents = nlp(text).ents
        if ents:
            return max(ents, key=lambda e: len(e.text)).label_
    return "OTHER"


def type_tag(text: str, doc=None, nlp=None) -> str:
    return surface_tag(text) or ner_tag(text, doc, nlp)


# ---------------------------------------------------------------- items

HOP_REF = re.compile(r"#(\d+)")


def hop_structure(decomp: list[dict]) -> tuple[list[list[int]], list[int], bool]:
    """depends_on (1-based) per hop, remaining_dep per hop, and whether the chain is linear.

    remaining_dep[k] = number of later hops that depend on hop k directly or transitively.
    Linear = every hop i>1 depends on hop i-1 (then remaining_dep[k] == K-k).
    """
    K = len(decomp)
    deps = [sorted({int(x) for x in HOP_REF.findall(d["question"])}) for d in decomp]
    children = {i: [j + 1 for j in range(K) if i in deps[j]] for i in range(1, K + 1)}
    remaining = []
    for k in range(1, K + 1):
        seen, stack = set(), list(children[k])
        while stack:
            c = stack.pop()
            if c not in seen:
                seen.add(c)
                stack.extend(children[c])
        remaining.append(len(seen))
    linear = all((i - 1) in deps[i - 1] for i in range(2, K + 1))
    return deps, remaining, linear


def stable_rng(item_id: str) -> random.Random:
    return random.Random(f"{SEED}|{item_id}")   # str seeds are deterministic across runs


def build_context(raw: dict) -> tuple[list[dict], list[dict]]:
    """Gold supporting paragraphs + N_DISTRACTORS fixed non-supporting ones, shuffled (seed 13)."""
    rng = stable_rng(raw["id"])
    paras = raw["paragraphs"]
    gold = [p for p in paras if p["is_supporting"]]
    gold_titles = {p["title"] for p in gold}
    pool, seen = [], set()
    for p in paras:
        if not p["is_supporting"] and p["title"] not in gold_titles and p["title"] not in seen:
            pool.append(p)
            seen.add(p["title"])
    distractors = rng.sample(pool, min(N_DISTRACTORS, len(pool)))
    ctx = [{"title": p["title"], "paragraph_text": p["paragraph_text"],
            "is_supporting": p["is_supporting"]} for p in gold + distractors]
    rng.shuffle(ctx)
    return ctx, distractors


def standalone_tags(texts: list[str], nlp) -> list[str]:
    """Type tag of each string on its own (no context), batched.

    Only an entity covering the whole string counts ("Treaty of Paris" is not GPE
    just because it contains "Paris"); a leading "the" is ignored.
    """
    out = []
    for t, doc in zip(texts, nlp.pipe(texts, batch_size=256)):
        s = surface_tag(t)
        if s is None:
            core = re.sub(r"^the\s+", "", t.strip(), flags=re.IGNORECASE).lower()
            whole = [e for e in doc.ents if re.sub(r"^the\s+", "", e.text, flags=re.IGNORECASE).lower() == core]
            s = whole[0].label_ if whole else "OTHER"
        out.append(s)
    return out


def distractor_candidates(distractors: list[dict], nlp) -> list[dict]:
    """Typed strings from the shown distractor paragraphs: titles, NER entities, years.

    A candidate is kept only if its in-context NER label agrees with the label of the
    string on its own (cheap guard against spaCy-sm mistags such as "Peter" = GPE).
    """
    raw, seen = [], set()

    def add(text, tag, source):
        text = text.strip()
        if text and text.lower() not in seen:
            seen.add(text.lower())
            raw.append({"text": text, "tag": tag, "source": source})

    for p in distractors:
        doc = nlp(p["paragraph_text"])
        add(p["title"], type_tag(p["title"], doc, nlp), "title")
        for ent in doc.ents:
            add(ent.text, surface_tag(ent.text) or ent.label_, "ner")
        for y in YEAR_IN_TEXT.findall(p["paragraph_text"]):
            add(y, "YEAR", "regex")
    alone = standalone_tags([c["text"] for c in raw], nlp)
    return [c for c, a in zip(raw, alone) if a == c["tag"]]


def prepare_item(raw: dict, split: str, nlp) -> dict:
    decomp = raw["question_decomposition"]
    deps, remaining, linear = hop_structure(decomp)
    paras = raw["paragraphs"]
    hops = []
    alone = standalone_tags([d["answer"] for d in decomp], nlp)
    for i, d in enumerate(decomp):
        support = paras[d["paragraph_support_idx"]]["paragraph_text"] if d.get("paragraph_support_idx") is not None else ""
        tag = type_tag(d["answer"], nlp(support) if support else None, nlp)
        hops.append({
            "question": d["question"], "answer": d["answer"], "tag": tag,
            "tag_confident": tag == alone[i],   # only confident answers enter the off_context pool
            "depends_on": deps[i], "remaining_dep": remaining[i],
        })
    ctx, distractors = build_context(raw)
    return {
        "id": raw["id"], "split": split, "composition": raw["id"].split("__")[0],
        "K": len(decomp), "question": raw["question"],
        "answer": raw["answer"], "answer_aliases": raw.get("answer_aliases", []),
        "hops": hops, "is_linear": linear, "context": ctx,
        "distractor_candidates": distractor_candidates(distractors, nlp),
    }


def items_path(split: str) -> Path:
    return ITEMS_DIR / f"musique_{split}_34hop.jsonl"


def prepare(splits=("dev", "train")) -> None:
    download()
    nlp = load_nlp()
    for split in splits:
        raws = [r for r in read_jsonl(raw_path(split))
                if r.get("answerable", True) and len(r["question_decomposition"]) in (3, 4)]
        print(f"[{split}] {len(raws)} answerable 3/4-hop items, tagging...")
        items = [prepare_item(r, split, nlp) for r in tqdm(raws, desc=split)]
        write_jsonl(items_path(split), items)
        print_counts(items, split)


def print_counts(items: list[dict], split: str) -> None:
    by_k = Counter(it["K"] for it in items)
    by_comp = Counter((it["composition"], it["is_linear"]) for it in items)
    tags = Counter(h["tag"] for it in items for h in it["hops"])
    print(f"\n=== {split}: {len(items)} items ===")
    for K in sorted(by_k):
        print(f"  {K}-hop: {by_k[K]}")
    print("  composition (linear?):")
    for (c, lin), n in sorted(by_comp.items()):
        print(f"    {c:8s} linear={lin!s:5s} {n}")
    print("  hop answer tags:", ", ".join(f"{t}={n}" for t, n in tags.most_common()))


def _order_key(item_id: str) -> str:
    return hashlib.sha256(f"{SEED}|{item_id}".encode()).hexdigest()


def load_items(splits, hops, limit_per_split_hop=None) -> list[dict]:
    """Prepared items in a fixed pseudo-random order; the limit takes the first N per split x K.

    The limit is an int (same for every K), a dict {K: int or None} (e.g. {3: 300, 4: None} to
    keep every 4-hop item), or None (all items).
    """
    out = []
    for split in splits:
        items = sorted(read_jsonl(items_path(split)), key=lambda it: _order_key(it["id"]))
        for K in hops:
            lim = limit_per_split_hop.get(K) if isinstance(limit_per_split_hop, dict) else limit_per_split_hop
            sel = [it for it in items if it["K"] == K]
            out.extend(sel[:lim] if lim else sel)
    return out
