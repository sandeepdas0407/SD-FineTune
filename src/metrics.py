"""Metrics for UC1 (support replies) and UC2 (banking triage JSON)."""
import json
import re
import statistics

from rouge_score import rouge_scorer
from sklearn.metrics import accuracy_score, f1_score

_PLACEHOLDER = re.compile(r"\{\{[^{}]+\}\}")
_STEP = re.compile(r"^\s*\d+[.)]\s", re.M)
_scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)


def uc1_metrics(rows):
    """rows: dicts with 'reference' and 'prediction'."""
    rl, ph_pred, steps_match, len_ratio = [], [], [], []
    for r in rows:
        ref, pred = r["reference"], r["prediction"]
        rl.append(_scorer.score(ref, pred)["rougeL"].fmeasure)
        ph_pred.append(bool(_PLACEHOLDER.search(pred)))
        steps_match.append(bool(_STEP.search(pred)) == bool(_STEP.search(ref)))
        len_ratio.append(len(pred.split()) / max(1, len(ref.split())))
    n = len(rows)
    ref_ph = sum(bool(_PLACEHOLDER.search(r["reference"])) for r in rows) / n
    return {
        "n": n,
        "rougeL_f1": round(statistics.mean(rl), 4),
        "placeholder_rate": round(sum(ph_pred) / n, 4),
        "placeholder_rate_reference": round(ref_ph, 4),
        "numbered_steps_agreement": round(sum(steps_match) / n, 4),
        "length_ratio_median": round(statistics.median(len_ratio), 3),
    }


def parse_uc2(text, labels):
    """-> (strict_json_ok, predicted_intent or None). Lenient fallback: regex for "intent": "..."."""
    strict = None
    try:
        obj = json.loads(text.strip())
        if isinstance(obj, dict):
            strict = obj
    except json.JSONDecodeError:
        pass
    if strict is not None:
        intent = strict.get("intent")
    else:
        m = re.search(r'"intent"\s*:\s*"([^"]+)"', text)
        intent = m.group(1) if m else None
    ok = strict is not None and isinstance(strict.get("intent"), str) and isinstance(strict.get("reply"), str)
    return ok, intent


def uc2_metrics(rows, labels):
    labels = set(labels)
    gold, pred, valid, in_set = [], [], 0, 0
    for r in rows:
        ok, intent = parse_uc2(r["prediction"], labels)
        valid += ok
        in_set += intent in labels
        gold.append(r["label"])
        pred.append(intent if intent in labels else "__invalid__")
    n = len(rows)
    return {
        "n": n,
        "intent_accuracy": round(accuracy_score(gold, pred), 4),
        "intent_macro_f1": round(f1_score(gold, pred, labels=sorted(labels), average="macro", zero_division=0), 4),
        "json_valid_rate": round(valid / n, 4),
        "label_in_set_rate": round(in_set / n, 4),
    }
