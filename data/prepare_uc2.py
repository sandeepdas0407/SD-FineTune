"""UC2: Banking77 queries (real) + synthetic replies -> JSON-target train/val/test JSONL."""
import csv
import json
import random
import urllib.request
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RAW = ROOT / "data" / "raw" / "banking77"
OUT = ROOT / "data" / "processed" / "uc2"
SYNTH = ROOT / "data" / "synth" / "banking77_replies.jsonl"
BASE = "https://raw.githubusercontent.com/PolyAI-LDN/task-specific-datasets/master/banking_data/"
N_TRAIN, N_VAL, N_TEST = 39, 3, 10  # per intent (77 intents -> 3003 / 231 / 770)
SEED = 42


def load(name):
    RAW.mkdir(parents=True, exist_ok=True)
    p = RAW / f"{name}.csv"
    if not p.exists():
        print("downloading", BASE + name + ".csv")
        urllib.request.urlretrieve(BASE + name + ".csv", p)
    return list(csv.DictReader(open(p, encoding="utf-8")))


def main():
    replies = defaultdict(list)
    for line in open(SYNTH, encoding="utf-8"):
        d = json.loads(line)
        replies[d["intent"]].append(d["reply"])
    train_raw, test_raw = load("train"), load("test")
    print("raw:", len(train_raw), "train /", len(test_raw), "test")

    rng = random.Random(SEED)
    norm = lambda s: " ".join(s.lower().split())
    by_tr, by_te = defaultdict(list), defaultdict(list)
    for r in train_raw:
        by_tr[r["category"]].append(r["text"].strip())
    seen_train = {norm(t) for t in (r["text"] for r in train_raw)}
    for r in test_raw:
        if norm(r["text"]) not in seen_train:  # drop exact overlaps with train
            by_te[r["category"]].append(r["text"].strip())

    labels = sorted(by_tr)
    assert set(labels) == set(replies), set(labels) ^ set(replies)
    assert set(by_te) <= set(labels)

    def row(split, i, text, intent):
        target = json.dumps({"intent": intent, "reply": rng.choice(replies[intent])}, ensure_ascii=False)
        return {"id": f"uc2-{split}-{i}", "input": text, "target": target, "label": intent}

    splits = {"train": [], "val": [], "test": []}
    for intent in labels:
        tr = sorted(set(by_tr[intent]))
        rng.shuffle(tr)
        te = sorted(set(by_te[intent]))
        rng.shuffle(te)
        for t in tr[:N_TRAIN]:
            splits["train"].append(row("train", len(splits["train"]), t, intent))
        for t in tr[N_TRAIN:N_TRAIN + N_VAL]:
            splits["val"].append(row("val", len(splits["val"]), t, intent))
        for t in te[:N_TEST]:
            splits["test"].append(row("test", len(splits["test"]), t, intent))

    OUT.mkdir(parents=True, exist_ok=True)
    for name, data in splits.items():
        rng.shuffle(data)
        with open(OUT / f"{name}.jsonl", "w", encoding="utf-8") as f:
            for d in data:
                f.write(json.dumps(d, ensure_ascii=False) + "\n")
        print(f"{name}: {len(data)} rows, {len({d['label'] for d in data})} intents")
    leak = {norm(d["input"]) for d in splits["train"]} & {norm(d["input"]) for d in splits["test"]}
    print("train/test overlap:", len(leak))
    assert not leak


if __name__ == "__main__":
    main()
