"""UC1: Bitext customer-support dataset -> stratified, de-duplicated train/val/test JSONL.

Near-duplicate handling: Bitext has many paraphrase variants of the same message. We normalise each
instruction (lowercase, strip {{placeholders}}, punctuation, whitespace) and split by that normalised
key, so variants of the same message never land in different splits.
"""
import csv
import json
import random
import re
import urllib.request
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RAW = ROOT / "data" / "raw" / "bitext.csv"
OUT = ROOT / "data" / "processed" / "uc1"
URL = ("https://huggingface.co/datasets/bitext/Bitext-customer-support-llm-chatbot-training-dataset/"
       "resolve/main/Bitext_Sample_Customer_Support_Training_Dataset_27K_responses-v11.csv")
N_TEST, N_VAL, N_TRAIN = 10, 8, 148  # per intent (27 intents -> 270 / 216 / ~4000)
SEED = 42


def norm(s: str) -> str:
    s = re.sub(r"\{\{[^}]*\}\}", " ", s.lower())
    s = re.sub(r"[^a-z0-9 ]+", " ", s)
    return re.sub(r"\s+", " ", s).strip()


def main():
    RAW.parent.mkdir(parents=True, exist_ok=True)
    if not RAW.exists():
        print("downloading", URL)
        urllib.request.urlretrieve(URL, RAW)
    rows = list(csv.DictReader(open(RAW, encoding="utf-8")))
    print("raw rows:", len(rows))

    # group by intent -> normalised key -> rows
    by_intent = defaultdict(lambda: defaultdict(list))
    for r in rows:
        if r["instruction"].strip() and r["response"].strip():
            by_intent[r["intent"]][norm(r["instruction"])].append(r)

    rng = random.Random(SEED)
    splits = {"train": [], "val": [], "test": []}
    for intent in sorted(by_intent):
        keys = sorted(by_intent[intent])
        rng.shuffle(keys)
        # one representative row per normalised key -> no exact/near duplicates anywhere
        reps = [rng.choice(by_intent[intent][k]) for k in keys]
        take = {"test": reps[:N_TEST], "val": reps[N_TEST:N_TEST + N_VAL],
                "train": reps[N_TEST + N_VAL:N_TEST + N_VAL + N_TRAIN]}
        for name, part in take.items():
            for r in part:
                splits[name].append({"id": f"uc1-{name}-{len(splits[name])}", "input": r["instruction"].strip(),
                                     "target": r["response"].strip(), "label": intent,
                                     "category": r["category"]})

    OUT.mkdir(parents=True, exist_ok=True)
    for name, data in splits.items():
        rng.shuffle(data)
        with open(OUT / f"{name}.jsonl", "w", encoding="utf-8") as f:
            for d in data:
                f.write(json.dumps(d, ensure_ascii=False) + "\n")
        print(f"{name}: {len(data)} rows, {len({d['label'] for d in data})} intents")

    # leakage check
    tr = {norm(d["input"]) for d in splits["train"]}
    leak = sum(norm(d["input"]) in tr for d in splits["test"] + splits["val"])
    print("near-duplicate leakage into val/test:", leak)
    assert leak == 0


if __name__ == "__main__":
    main()
