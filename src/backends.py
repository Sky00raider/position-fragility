"""Generation backends behind one interface.

    backend.generate(prompt_ids, seeds, max_tokens) -> list[Gen]

Prompts arrive as token ids (built in prompts.py), so both backends see exactly the
same input. Only the `backend:` key in the config decides which one runs.
"""
from __future__ import annotations

from dataclasses import dataclass

from .config import Config


@dataclass
class Gen:
    text: str
    n_gen_tokens: int
    finish_reason: str   # "stop" | "length"


def load_tokenizer(cfg: Config):
    from transformers import AutoTokenizer
    return AutoTokenizer.from_pretrained(cfg.tokenizer_id or cfg.model_id, revision=cfg.revision)


class Backend:
    def generate(self, prompt_ids: list[list[int]], seeds: list[int], max_tokens: int) -> list[Gen]:
        raise NotImplementedError

    def info(self) -> dict:
        return {}


class VLLMBackend(Backend):
    def __init__(self, cfg: Config):
        from vllm import LLM
        self.cfg = cfg
        self.llm = LLM(
            model=cfg.model_id, tokenizer=cfg.tokenizer_id or cfg.model_id, revision=cfg.revision,
            max_model_len=cfg.max_model_len, gpu_memory_utilization=cfg.gpu_memory_utilization,
            seed=cfg.seed,
        )

    def generate(self, prompt_ids, seeds, max_tokens):
        from vllm import SamplingParams
        from vllm.inputs import TokensPrompt
        d = self.cfg.decoding
        prompts, params = [], []
        for ids, s in zip(prompt_ids, seeds):
            room = self.cfg.max_model_len - len(ids)
            params.append(SamplingParams(
                temperature=d.temperature, top_p=d.top_p, top_k=d.top_k if d.top_k > 0 else -1,
                max_tokens=max(1, min(max_tokens, room)), seed=s,
            ))
            prompts.append(TokensPrompt(prompt_token_ids=ids))
        outs = self.llm.generate(prompts, params, use_tqdm=False)
        return [Gen(o.outputs[0].text, len(o.outputs[0].token_ids),
                    "length" if o.outputs[0].finish_reason == "length" else "stop") for o in outs]

    def info(self):
        import vllm
        return {"backend": "vllm", "lib_version": f"vllm {vllm.__version__}"}


class HFBackend(Backend):
    """transformers generate, one request at a time (CPU dev box, or GPU fallback)."""

    def __init__(self, cfg: Config):
        import torch
        from transformers import AutoModelForCausalLM
        self.cfg = cfg
        self.torch = torch
        self.tok = load_tokenizer(cfg)
        dtype = torch.float32 if cfg.device == "cpu" else torch.bfloat16
        self.model = AutoModelForCausalLM.from_pretrained(
            cfg.model_id, revision=cfg.revision, torch_dtype=dtype,
        ).to(cfg.device).eval()

    def generate(self, prompt_ids, seeds, max_tokens):
        torch, d = self.torch, self.cfg.decoding
        gens = []
        for ids, s in zip(prompt_ids, seeds):
            torch.manual_seed(s)
            room = self.cfg.max_model_len - len(ids)
            n_max = max(1, min(max_tokens, room))
            x = torch.tensor([ids], device=self.cfg.device)
            with torch.no_grad():
                out = self.model.generate(
                    x, attention_mask=torch.ones_like(x), do_sample=True,
                    temperature=d.temperature, top_p=d.top_p, top_k=d.top_k,
                    max_new_tokens=n_max, pad_token_id=self.tok.pad_token_id or self.tok.eos_token_id,
                )
            new = out[0, len(ids):].tolist()
            eos_ids = self.model.generation_config.eos_token_id
            eos_ids = set(eos_ids if isinstance(eos_ids, list) else [eos_ids])
            hit_eos = bool(new) and new[-1] in eos_ids
            gens.append(Gen(self.tok.decode(new, skip_special_tokens=True), len(new),
                            "stop" if hit_eos or len(new) < n_max else "length"))
        return gens

    def info(self):
        import transformers
        return {"backend": "hf", "lib_version": f"transformers {transformers.__version__}"}


def make_backend(cfg: Config) -> Backend:
    return VLLMBackend(cfg) if cfg.backend == "vllm" else HFBackend(cfg)
