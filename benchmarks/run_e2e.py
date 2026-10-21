"""End-to-end Qwen2.5 inference benchmark.

Compares eager fp16, torch.compile, bitsandbytes NF4, and TritonForge on:
  TTFT (prefill), inter-token latency, decode throughput, peak VRAM, and
  wikitext-2 perplexity.

    PYTHONPATH=src python benchmarks/run_e2e.py

Writes results/e2e.json.
"""

from __future__ import annotations

import gc
import json
import math
import os
import pathlib
import statistics
import time
import traceback

import torch

from tritonforge import env

MODEL_ID = os.environ.get("MODEL_ID", "Qwen/Qwen2.5-0.5B-Instruct")
MAX_NEW = int(os.environ.get("MAX_NEW", "64"))
PPL_SEQ_LEN = int(os.environ.get("PPL_SEQ_LEN", "512"))
PPL_MAX_SEQS = int(os.environ.get("PPL_MAX_SEQS", "16"))
GROUP_SIZE = int(os.environ.get("GROUP_SIZE", "128"))
PROMPT = "Explain how tensor cores differ from CUDA cores on an NVIDIA GPU."


def load_tokenizer():
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(MODEL_ID)


def load_fp16():
    from transformers import AutoModelForCausalLM

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, torch_dtype=torch.float16, device_map="cuda"
    )
    model.eval()
    return model


def load_bnb_nf4():
    from transformers import AutoModelForCausalLM, BitsAndBytesConfig

    cfg = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=False,
    )
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, quantization_config=cfg, device_map="cuda"
    )
    model.eval()
    return model


def load_tritonforge():
    from tritonforge.integration import patch_model

    model = load_fp16()
    t0 = time.time()
    patch_model(model, group_size=GROUP_SIZE)
    model._patch_seconds = time.time() - t0
    return model


def load_compiled():
    model = load_fp16()
    model = torch.compile(model, mode="default")
    return model


def _event() -> torch.cuda.Event:
    return torch.cuda.Event(enable_timing=True)


@torch.no_grad()
def bench_generation(model, tokenizer) -> dict:
    device = "cuda"
    ids = tokenizer(PROMPT, return_tensors="pt").input_ids.to(device)
    max_new = MAX_NEW

    def _step(prev, past):
        out = model(prev, past_key_values=past, use_cache=True)
        nxt = out.logits[:, -1, :].argmax(-1, keepdim=True)
        return nxt, out.past_key_values

    for _ in range(2):
        warm = model(ids, use_cache=True)
    prev = warm.logits[:, -1, :].argmax(-1, keepdim=True)
    past = warm.past_key_values
    for _ in range(5):
        prev, past = _step(prev, past)
    torch.cuda.synchronize()

    torch.cuda.reset_peak_memory_stats()
    start, end = _event(), _event()
    start.record()
    out = model(ids, use_cache=True)
    end.record()
    torch.cuda.synchronize()
    ttft_ms = start.elapsed_time(end)

    nxt = out.logits[:, -1, :].argmax(-1, keepdim=True)
    past = out.past_key_values
    tokens = [int(nxt.item())]
    step_ms: list[float] = []
    for _ in range(max_new - 1):
        s, e = _event(), _event()
        s.record()
        out = model(nxt, past_key_values=past, use_cache=True)
        e.record()
        torch.cuda.synchronize()
        step_ms.append(s.elapsed_time(e))
        past = out.past_key_values
        nxt = out.logits[:, -1, :].argmax(-1, keepdim=True)
        tokens.append(int(nxt.item()))

    peak_gb = torch.cuda.max_memory_allocated() / 1e9
    decode_total = sum(step_ms)
    text = tokenizer.decode(tokens, skip_special_tokens=True)
    return {
        "ttft_ms": ttft_ms,
        "itl_ms_mean": statistics.fmean(step_ms),
        "itl_ms_p50": statistics.median(step_ms),
        "itl_ms_std": statistics.pstdev(step_ms) if len(step_ms) > 1 else 0.0,
        "decode_tokens_per_s": (max_new - 1) * 1000.0 / decode_total if decode_total else None,
        "e2e_tokens_per_s": max_new * 1000.0 / (ttft_ms + decode_total),
        "peak_vram_gb": peak_gb,
        "generated_tokens": tokens,
        "sample_text": text,
    }


@torch.no_grad()
def perplexity(model, encodings: torch.Tensor):
    total_nll = 0.0
    total_tok = 0
    n_seqs = min(encodings.shape[1] // PPL_SEQ_LEN, PPL_MAX_SEQS)
    for i in range(n_seqs):
        chunk = encodings[:, i * PPL_SEQ_LEN : (i + 1) * PPL_SEQ_LEN].cuda()
        out = model(chunk, labels=chunk)
        total_nll += float(out.loss) * (PPL_SEQ_LEN - 1)
        total_tok += PPL_SEQ_LEN - 1
    return math.exp(total_nll / total_tok), n_seqs


def build_wikitext(tokenizer) -> torch.Tensor:
    from datasets import load_dataset

    ds = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split="test")
    text = "\n\n".join(t for t in ds["text"] if t.strip())
    return tokenizer(text, return_tensors="pt").input_ids


def run_variant(name: str, build, tokenizer, encodings) -> dict:
    result: dict = {"name": name, "status": "ok"}
    model = None
    try:
        t0 = time.time()
        model = build()
        result["load_seconds"] = time.time() - t0
        result.update(bench_generation(model, tokenizer))
        ppl, n_seqs = perplexity(model, encodings)
        result["perplexity"] = ppl
        result["ppl_sequences"] = n_seqs
    except Exception as exc:  # noqa: BLE001
        result["status"] = "error"
        result["error"] = f"{type(exc).__name__}: {exc}"
        result["traceback"] = traceback.format_exc()[-2000:]
    finally:
        if model is not None:
            del model
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
    return result


def main() -> None:
    env.require_cuda()
    print(env.describe())
    tokenizer = load_tokenizer()
    encodings = build_wikitext(tokenizer)
    print(f"wikitext tokens: {encodings.shape[1]}")

    variants = [
        ("eager_fp16", load_fp16),
        ("tritonforge", load_tritonforge),
        ("bnb_nf4", load_bnb_nf4),
        ("torch_compile", load_compiled),
    ]
    results = []
    for name, build in variants:
        print(f"--- {name} ---")
        r = run_variant(name, build, tokenizer, encodings)
        results.append(r)
        if r["status"] == "ok":
            print(
                f"  ttft={r['ttft_ms']:.1f}ms itl={r['itl_ms_mean']:.2f}ms "
                f"decode={r['decode_tokens_per_s']:.1f} tok/s ppl={r['perplexity']:.4f} "
                f"vram={r['peak_vram_gb']:.2f}GB"
            )
        else:
            print(f"  FAILED: {r['error']}")

    eager_ppl = next((r.get("perplexity") for r in results if r["name"] == "eager_fp16"), None)
    for r in results:
        if r["status"] == "ok" and eager_ppl:
            r["ppl_delta_vs_fp16"] = r["perplexity"] - eager_ppl

    out = pathlib.Path("results")
    out.mkdir(parents=True, exist_ok=True)
    payload = {
        "model_id": MODEL_ID,
        "max_new_tokens": MAX_NEW,
        "group_size": GROUP_SIZE,
        "device": env.describe(),
        "results": results,
    }
    (out / "e2e.json").write_text(json.dumps(payload, indent=2))
    print(f"wrote {out / 'e2e.json'}")


if __name__ == "__main__":
    main()
