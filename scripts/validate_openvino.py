#!/usr/bin/env python
"""Numerical parity and perplexity for every OpenVINO variant against PyTorch fp32.

Metrics per variant:
  * max / mean absolute logit error and a relative error at the last prompt token
  * top-1 agreement and top-5 overlap over 65 prompts
  * mean KL divergence against the fp32 softmax
  * wikitext-2 test perplexity on a fixed number of 512-token windows

The PyTorch reference is cached under ``artifacts/x/`` so adding a variant does
not recompute it.

Usage:
    python scripts/validate_openvino.py --models /content/models \
        --out results/x/validation.json --n-windows 8
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
import pathlib
import time

import torch
import torch.nn.functional as F

ROOT = pathlib.Path(__file__).resolve().parents[1]
MODEL_ID = "Qwen/Qwen2.5-0.5B-Instruct"


def load_prompts(path: pathlib.Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def prompts_sha(prompts: list[dict]) -> str:
    joined = "\n".join(item["prompt"] for item in prompts)
    return hashlib.sha256(joined.encode()).hexdigest()[:16]


def wikitext_windows(tokenizer, n_windows: int, seq_len: int) -> torch.Tensor:
    from datasets import load_dataset

    dataset = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split="test")
    ids: list[int] = []
    for row in dataset:
        if row["text"].strip():
            ids.extend(tokenizer(row["text"], add_special_tokens=False)["input_ids"])
    need = n_windows * seq_len
    if len(ids) < need:
        raise RuntimeError(f"wikitext-2 test split only has {len(ids)} tokens, need {need}")
    return torch.tensor([ids[i : i + seq_len] for i in range(0, need, seq_len)])


def last_token_logits(model, enc) -> torch.Tensor:
    with torch.no_grad():
        out = model(**enc)
    return out.logits[0, -1].float()


def compute_reference(prompts: list[dict], tokenizer, windows, cache: pathlib.Path) -> dict:
    from transformers import AutoModelForCausalLM

    if cache.exists():
        payload = torch.load(cache, map_location="cpu", weights_only=False)
        if (
            payload.get("prompts_sha") == prompts_sha(prompts)
            and payload.get("n_windows") == len(windows)
            and payload.get("seq_len") == windows.shape[1]
        ):
            print(f"[reference] reusing {cache}")
            return payload

    print("[reference] loading PyTorch fp32 model", flush=True)
    model = AutoModelForCausalLM.from_pretrained(MODEL_ID, dtype=torch.float32)
    model.eval()
    logits = []
    start = time.perf_counter()
    for item in prompts:
        enc = tokenizer(item["prompt"], return_tensors="pt")
        logits.append(last_token_logits(model, enc))
    logits_s = time.perf_counter() - start

    nll = 0.0
    n_tokens = 0
    start = time.perf_counter()
    with torch.no_grad():
        for window in windows:
            out = model(input_ids=window.unsqueeze(0))
            shift_logits = out.logits[:, :-1, :].float()
            shift_labels = window[1:].unsqueeze(0)
            nll += F.cross_entropy(
                shift_logits.reshape(-1, shift_logits.shape[-1]),
                shift_labels.reshape(-1),
                reduction="sum",
            ).item()
            n_tokens += shift_labels.numel()
    ppl_s = time.perf_counter() - start
    payload = {
        "prompts_sha": prompts_sha(prompts),
        "n_windows": len(windows),
        "seq_len": windows.shape[1],
        "logits": torch.stack(logits),
        "ppl": math.exp(nll / n_tokens),
        "ppl_nll": nll,
        "ppl_tokens": n_tokens,
        "ppl_seconds": ppl_s,
        "logits_seconds": logits_s,
    }
    cache.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, cache)
    del model
    gc.collect()
    return payload


def compare(ref: torch.Tensor, cand: torch.Tensor) -> dict:
    diff = (cand - ref).abs()
    denom = ref.abs().clamp_min(1e-5)
    top1_ref = ref.argmax(dim=-1)
    top1_cand = cand.argmax(dim=-1)
    top5_ref = ref.topk(5, dim=-1).indices
    top5_cand = cand.topk(5, dim=-1).indices
    overlap = [
        len(set(a.tolist()) & set(b.tolist())) / 5.0
        for a, b in zip(top5_ref, top5_cand, strict=True)
    ]
    logp_ref = F.log_softmax(ref, dim=-1)
    logp_cand = F.log_softmax(cand, dim=-1)
    kl = (logp_ref.exp() * (logp_ref - logp_cand)).sum(dim=-1)
    return {
        "max_abs_err": diff.max().item(),
        "mean_abs_err": diff.mean().item(),
        "max_rel_err": (diff / denom).max().item(),
        "top1_agreement_pct": 100.0 * (top1_ref == top1_cand).float().mean().item(),
        "mean_top5_overlap": sum(overlap) / len(overlap),
        "mean_kl": kl.mean().item(),
    }


def eval_variant(path: pathlib.Path, prompts, tokenizer, windows, ref_logits, ref_ppl):
    from optimum.intel import OVModelForCausalLM

    print(f"[{path.name}] loading", flush=True)
    model = OVModelForCausalLM.from_pretrained(
        str(path),
        ov_config={"PERFORMANCE_HINT": "LATENCY", "INFERENCE_NUM_THREADS": "2"},
    )
    logits = []
    start = time.perf_counter()
    for item in prompts:
        enc = tokenizer(item["prompt"], return_tensors="pt")
        try:
            model.request.reset_state()
        except AttributeError:
            pass
        logits.append(last_token_logits(model, enc))
    logits_s = time.perf_counter() - start
    cand = torch.stack(logits)
    parity = compare(ref_logits, cand)

    nll = 0.0
    n_tokens = 0
    start = time.perf_counter()
    for window in windows:
        try:
            model.request.reset_state()
        except AttributeError:
            pass
        with torch.no_grad():
            out = model(input_ids=window.unsqueeze(0))
        shift_logits = out.logits[:, :-1, :].float()
        shift_labels = window[1:].unsqueeze(0)
        nll += F.cross_entropy(
            shift_logits.reshape(-1, shift_logits.shape[-1]),
            shift_labels.reshape(-1),
            reduction="sum",
        ).item()
        n_tokens += shift_labels.numel()
    ppl_s = time.perf_counter() - start
    ppl = math.exp(nll / n_tokens)

    result = {
        **parity,
        "ppl": ppl,
        "ppl_delta_pct": 100.0 * (ppl - ref_ppl) / ref_ppl,
        "ppl_tokens": n_tokens,
        "ppl_seconds": ppl_s,
        "ppl_tokens_per_s": n_tokens / ppl_s,
        "logits_seconds": logits_s,
        "per_prompt_top1": [
            bool(a == b)
            for a, b in zip(ref_logits.argmax(-1), cand.argmax(-1), strict=True)
        ],
    }
    del model
    gc.collect()
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models", default="/content/models")
    parser.add_argument("--models-json", default=str(ROOT / "results/x/models.json"))
    parser.add_argument("--prompts", default=str(ROOT / "data/parity_prompts.jsonl"))
    parser.add_argument("--out", default=str(ROOT / "results/x/validation.json"))
    parser.add_argument("--cache", default=str(ROOT / "artifacts/x/reference.pt"))
    parser.add_argument("--n-windows", type=int, default=8)
    parser.add_argument("--seq-len", type=int, default=512)
    parser.add_argument("--variants", nargs="*", default=None)
    args = parser.parse_args()

    from transformers import AutoTokenizer

    torch.set_num_threads(2)
    prompts = load_prompts(pathlib.Path(args.prompts))
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    windows = wikitext_windows(tokenizer, args.n_windows, args.seq_len)
    print(
        f"{len(prompts)} prompts; wikitext-2: {args.n_windows} windows x {args.seq_len} "
        f"= {windows.numel()} tokens",
        flush=True,
    )

    hw = json.loads((ROOT / "results/x/hardware.json").read_text())["cpu"]["model_name"]
    ref = compute_reference(prompts, tokenizer, windows, pathlib.Path(args.cache))
    ref_logits, ref_ppl = ref["logits"], ref["ppl"]

    models_json = json.loads(pathlib.Path(args.models_json).read_text())
    names = args.variants or [row["variant"] for row in models_json["variants"]]
    rows = []
    for name in names:
        path = pathlib.Path(args.models) / f"ov_{name}"
        if not path.exists():
            print(f"[{name}] missing at {path}, skipping")
            continue
        metrics = eval_variant(path, prompts, tokenizer, windows, ref_logits, ref_ppl)
        rows.append({"variant": name, **metrics})
        print(
            f"[{name}] top1={metrics['top1_agreement_pct']:.1f}% "
            f"max_abs={metrics['max_abs_err']:.3f} ppl={metrics['ppl']:.3f} "
            f"delta={metrics['ppl_delta_pct']:+.2f}%",
            flush=True,
        )

    payload = {
        "model_id": MODEL_ID,
        "hardware_cpu": hw,
        "note": "Generated by scripts/validate_openvino.py against a PyTorch fp32 oracle.",
        "prompts": len(prompts),
        "wikitext": {
            "split": "test",
            "n_windows": args.n_windows,
            "seq_len": args.seq_len,
            "tokens_scored": int(windows.numel() - args.n_windows),
        },
        "reference": {
            "framework": "PyTorch fp32",
            "ppl": ref_ppl,
            "ppl_seconds": ref["ppl_seconds"],
            "logits_seconds": ref["logits_seconds"],
            "ppl_tokens": ref["ppl_tokens"],
        },
        "variants": rows,
    }
    out = pathlib.Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2) + "\n")

    print(
        f"\n{'variant':18s} {'top1 %':>7s} {'max abs':>9s} {'mean abs':>9s} "
        f"{'ppl':>8s} {'dPPL %':>8s}"
    )
    for row in rows:
        print(
            f"{row['variant']:18s} {row['top1_agreement_pct']:7.1f} {row['max_abs_err']:9.3f} "
            f"{row['mean_abs_err']:9.4f} {row['ppl']:8.3f} {row['ppl_delta_pct']:+8.2f}"
        )
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
