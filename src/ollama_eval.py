"""Evaluate the local Ollama models (Q8_0 GGUF, CPU) on the held-out test splits and compare with the GPU run.

  python src/ollama_eval.py            # UC2: all 770; UC1: 100 stratified (seeded)
Resumable: predictions are appended to results/ollama/<uc>/predictions_<variant>.jsonl and finished ids are skipped.
Writes results/ollama/metrics.json and prints a comparison table.
"""
import json
import random
import sys
import time
import urllib.request
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from metrics import uc1_metrics, uc2_metrics  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "results" / "ollama"
URL = "http://127.0.0.1:11434/api/chat"
PLAN = {  # uc: (models {variant: ollama name}, max tokens = GPU eval's max_new_tokens, sample size or None)
    "uc2": ({"base": "sd-banking-base", "finetuned": "sd-banking-ft"}, 150, None),
    "uc1": ({"base": "sd-support-base", "finetuned": "sd-support-ft"}, 300, 100),
}


def read_jsonl(p):
    return [json.loads(l) for l in open(p, encoding="utf-8") if l.strip()] if p.exists() else []


def stratified(rows, n, seed=42):
    """Round-robin across intents so every intent is represented."""
    rng = random.Random(seed)
    by = defaultdict(list)
    for r in rows:
        by[r["label"]].append(r)
    for v in by.values():
        rng.shuffle(v)
    out, i = [], 0
    while len(out) < n:
        for k in sorted(by):
            if i < len(by[k]) and len(out) < n:
                out.append(by[k][i])
        i += 1
    return out


def chat(model, text, max_tokens):
    body = json.dumps({"model": model, "stream": False, "keep_alive": "30m",
                       "options": {"num_predict": max_tokens},
                       "messages": [{"role": "user", "content": text}]}).encode()
    req = urllib.request.Request(URL, body, {"Content-Type": "application/json"})
    for attempt in range(4):  # a stalled request (sleep, memory pressure) is retried, not fatal
        try:
            with urllib.request.urlopen(req, timeout=180) as r:
                return json.load(r)["message"]["content"].strip()
        except (TimeoutError, OSError) as e:
            print(f"  attempt {attempt + 1}/4 failed ({type(e).__name__}: {e})", flush=True)
            if attempt < 3:
                time.sleep(10)
    raise RuntimeError(f"Ollama did not answer after 4 attempts ({model})")


def main():
    labels = json.loads((ROOT / "data" / "synth" / "banking77_labels.json").read_text(encoding="utf-8"))
    summary = {}
    for uc, (models, max_tok, n) in PLAN.items():
        test = read_jsonl(ROOT / "data" / "processed" / uc / "test.jsonl")
        test = stratified(test, n) if n else test
        (OUT / uc).mkdir(parents=True, exist_ok=True)
        for variant, model in models.items():
            pf = OUT / uc / f"predictions_{variant}.jsonl"
            done = {r["id"] for r in read_jsonl(pf)}
            todo = [r for r in test if r["id"] not in done]
            t0 = time.time()
            with open(pf, "a", encoding="utf-8") as f:
                for k, r in enumerate(todo, 1):
                    pred = chat(model, r["input"], max_tok)
                    f.write(json.dumps({"id": r["id"], "input": r["input"], "reference": r["target"],
                                        "label": r["label"], "prediction": pred}, ensure_ascii=False) + "\n")
                    f.flush()
                    if k % 25 == 0 or k == len(todo):
                        el = time.time() - t0
                        print(f"[{uc} {variant}] {len(done) + k}/{len(test)}  {el / k:.1f}s/msg  "
                              f"eta {el / k * (len(todo) - k) / 60:.0f} min", flush=True)
        ids = {r["id"] for r in test}
        summary[uc] = {"n": len(test)}
        for variant in models:
            local = [r for r in read_jsonl(OUT / uc / f"predictions_{variant}.jsonl") if r["id"] in ids]
            gpu = [r for r in read_jsonl(ROOT / "results" / uc / f"predictions_{variant}.jsonl") if r["id"] in ids]
            m = uc1_metrics if uc == "uc1" else (lambda rows: uc2_metrics(rows, labels))
            summary[uc][variant] = {"ollama_cpu_q8": m(local), "gpu_fp16_same_ids": m(gpu)}
    (OUT / "metrics.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
