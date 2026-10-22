"""Day-1 GPU check: load the configured model, generate for 10 real prompts, report speed.

    python scripts/gpu_smoke.py --config configs/qwen3_8b_nothink.yaml

Prints GPU, library versions, batched and single-stream tokens/s, VRAM, truncation rate,
and one full output so you can eyeball the hop format. Needs data/items/ (run prepare-data first).
"""
from __future__ import annotations

import argparse
import platform
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.backends import load_tokenizer, make_backend  # noqa: E402
from src.config import hf_token_present, load_config  # noqa: E402
from src.data import load_items  # noqa: E402
from src.parse import extract_final, hops_well_formed, parse_hops, split_think  # noqa: E402
from src.prompts import build_ids, user_message  # noqa: E402


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--n", type=int, default=10)
    args = ap.parse_args()
    cfg = load_config(args.config)

    import torch
    print(f"python {platform.python_version()} | torch {torch.__version__} | cuda {torch.version.cuda}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)} | capability {torch.cuda.get_device_capability(0)}")
    else:
        print("GPU: none visible to torch")
    print(f"HF_TOKEN set: {hf_token_present()}")
    print(f"model {cfg.model_id} | backend {cfg.backend} | mode {cfg.mode}")

    items = load_items(splits=["dev"], hops=[3, 4], limit_per_split_hop=args.n // 2 + 1)[: args.n]
    tok = load_tokenizer(cfg)
    ids = [build_ids(tok, cfg.model_id, user_message(it), cfg.think) for it in items]
    print(f"prompt tokens: mean {sum(map(len, ids)) / len(ids):.0f}, max {max(map(len, ids))}")

    t0 = time.time()
    backend = make_backend(cfg)
    print(f"load time: {time.time() - t0:.1f}s | {backend.info()}")

    mt = cfg.decoding.max_tokens
    t0 = time.time()
    gens = backend.generate(ids[:1], [1], mt)
    dt = time.time() - t0
    print(f"single-stream: {gens[0].n_gen_tokens} tokens in {dt:.1f}s = {gens[0].n_gen_tokens / dt:.1f} tok/s")

    t0 = time.time()
    gens = backend.generate(ids, list(range(len(ids))), mt)
    dt = time.time() - t0
    n_tok = sum(g.n_gen_tokens for g in gens)
    print(f"batched x{len(ids)}: {n_tok} tokens in {dt:.1f}s = {n_tok / dt:.1f} tok/s")
    print(f"mean gen tokens {n_tok / len(gens):.0f} | truncated {sum(g.finish_reason == 'length' for g in gens)}/{len(gens)}")

    # format compliance: K well-formed hop lines (after </think> in think mode) + a final answer
    ok_hops = ok_final = ok_think_hops = 0
    for it, g in zip(items, gens):
        think_part, answer_part, _ = split_think(g.text, cfg.think)
        ok_hops += hops_well_formed(parse_hops(answer_part), it["K"])
        ok_think_hops += hops_well_formed(parse_hops(think_part), it["K"])
        ok_final += extract_final(answer_part) is not None
    print(f"format: K hop lines {ok_hops}/{len(gens)} | final answer {ok_final}/{len(gens)}"
          + (f" | K hop lines inside <think> {ok_think_hops}/{len(gens)}" if cfg.think else ""))

    if torch.cuda.is_available():
        free, total = torch.cuda.mem_get_info()
        print(f"VRAM used {(total - free) / 2**30:.1f} / {total / 2**30:.1f} GiB")

    print("\n===== example output =====")
    print(items[0]["question"])
    print(gens[0].text)


if __name__ == "__main__":
    main()
