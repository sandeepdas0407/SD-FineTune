"""Batched greedy generation for: base (zero-shot), base_fewshot (3-shot), finetuned (base + LoRA adapter).

Usage:
  python src/evaluate.py --config configs/uc2.yaml --variants base,base_fewshot
  python src/evaluate.py --config configs/uc2.yaml --variants finetuned --adapter outputs/uc2/adapter
Writes <results_dir>/predictions_<variant>.jsonl
"""
import argparse
import json
import random
import sys
import time
from pathlib import Path

import torch
import yaml
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parent))
from prompts import build_messages  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent


def read_jsonl(p):
    return [json.loads(l) for l in open(p, encoding="utf-8")]


def pick_fewshot(train, k, seed=0):
    """k examples with distinct labels, fixed seed -> identical shots for every test item."""
    rng = random.Random(seed)
    pool = train[:]
    rng.shuffle(pool)
    shots, seen = [], set()
    for r in pool:
        if r["label"] not in seen:
            shots.append(r)
            seen.add(r["label"])
        if len(shots) == k:
            break
    return shots


def load_model(model_id, dtype, adapter=None):
    model = AutoModelForCausalLM.from_pretrained(model_id, dtype=dtype, attn_implementation="sdpa")
    if adapter:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, adapter).merge_and_unload()
    return model.to("cuda" if torch.cuda.is_available() else "cpu").eval()


@torch.no_grad()
def generate(model, tok, prompts, max_new_tokens, batch_size):
    outs = []
    for i in range(0, len(prompts), batch_size):
        batch = prompts[i:i + batch_size]
        enc = tok(batch, return_tensors="pt", padding=True).to(model.device)
        gen = model.generate(**enc, max_new_tokens=max_new_tokens, do_sample=False,
                             pad_token_id=tok.pad_token_id)
        outs += tok.batch_decode(gen[:, enc["input_ids"].shape[1]:], skip_special_tokens=True)
        print(f"  {min(i + batch_size, len(prompts))}/{len(prompts)}", flush=True)
    return [o.strip() for o in outs]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--variants", default="base,base_fewshot")
    ap.add_argument("--adapter", help="LoRA adapter dir (for 'finetuned')")
    ap.add_argument("--model", help="override base model id")
    ap.add_argument("--limit", type=int)
    ap.add_argument("--results_dir")
    a = ap.parse_args()

    cfg = yaml.safe_load(open(a.config))
    e = cfg["eval"]
    model_id = a.model or cfg["base_model"]
    data = ROOT / cfg["data_dir"]
    res = Path(a.results_dir or ROOT / cfg["results_dir"])
    res.mkdir(parents=True, exist_ok=True)
    test = read_jsonl(data / "test.jsonl")[: a.limit] if a.limit else read_jsonl(data / "test.jsonl")
    shots = pick_fewshot(read_jsonl(data / "train.jsonl"), e["fewshot_k"])
    cuda = torch.cuda.is_available()
    dtype = getattr(torch, e["dtype"]) if cuda else torch.float32

    tok = AutoTokenizer.from_pretrained(model_id, padding_side="left")
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    variants = a.variants.split(",")
    base = None
    for v in variants:
        t0 = time.time()
        if v == "finetuned":
            assert a.adapter, "--adapter is required for the finetuned variant"
            model = load_model(model_id, dtype, a.adapter)
        else:
            base = base or load_model(model_id, dtype)
            model = base
        fs = shots if v == "base_fewshot" else None
        prompts = [tok.apply_chat_template(build_messages(cfg["task"], r["input"], fs), tokenize=False,
                                           add_generation_prompt=True) for r in test]
        print(f"[{v}] {len(prompts)} prompts")
        preds = generate(model, tok, prompts, cfg["max_new_tokens"], e["batch_size"])
        with open(res / f"predictions_{v}.jsonl", "w", encoding="utf-8") as f:
            for r, p in zip(test, preds):
                f.write(json.dumps({"id": r["id"], "input": r["input"], "reference": r["target"],
                                    "label": r["label"], "prediction": p}, ensure_ascii=False) + "\n")
        print(f"[{v}] done in {time.time() - t0:.0f}s -> {res / f'predictions_{v}.jsonl'}")
        if v == "finetuned":
            del model
            if cuda:
                torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
