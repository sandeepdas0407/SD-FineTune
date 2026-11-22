"""Build results/metrics.json, results/report.html, and a blind A/B sample for the Claude judge.

Usage: python src/make_report.py
"""
import html
import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from metrics import uc1_metrics, uc2_metrics  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
RES = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "results"
VARIANTS = ["base", "base_fewshot", "finetuned"]
NAMES = {"base": "Base (zero-shot)", "base_fewshot": "Base (3-shot)", "finetuned": "Fine-tuned (LoRA)"}
UCS = {
    "uc1": ("UC1 · Customer-support replies (Bitext)", ["rougeL_f1", "placeholder_rate", "numbered_steps_agreement", "length_ratio_median"]),
    "uc2": ("UC2 · Banking triage JSON (Banking77 + synthetic replies)", ["intent_accuracy", "intent_macro_f1", "json_valid_rate", "label_in_set_rate"]),
}


def load(uc, v):
    p = RES / uc / f"predictions_{v}.jsonl"
    return [json.loads(l) for l in open(p, encoding="utf-8")] if p.exists() else None


def judge_sample(rows_base, rows_ft, n=30, seed=7):
    """Blind A/B pairs: base zero-shot vs fine-tuned, order randomised; key kept separately."""
    rng = random.Random(seed)
    idx = rng.sample(range(len(rows_ft)), min(n, len(rows_ft)))
    pairs, key = [], {}
    for k, i in enumerate(idx):
        a, b = (rows_base[i], rows_ft[i]) if rng.random() < 0.5 else (rows_ft[i], rows_base[i])
        pairs.append({"pair": k, "customer_message": rows_ft[i]["input"], "A": a["prediction"], "B": b["prediction"]})
        key[k] = "A" if a is rows_ft[i] else "B"
    (RES / "uc1" / "judge_pairs.jsonl").write_text("\n".join(json.dumps(p, ensure_ascii=False) for p in pairs), encoding="utf-8")
    (RES / "uc1" / "judge_key.json").write_text(json.dumps(key), encoding="utf-8")


def main():
    labels = json.loads((ROOT / "data" / "synth" / "banking77_labels.json").read_text(encoding="utf-8"))
    allm, sections = {}, []
    for uc, (title, keys) in UCS.items():
        preds = {v: load(uc, v) for v in VARIANTS}
        m = {v: (uc1_metrics(r) if uc == "uc1" else uc2_metrics(r, labels)) for v, r in preds.items() if r}
        if not m:
            continue
        allm[uc] = m
        head = "".join(f"<th>{NAMES[v]}</th>" for v in m)
        body = ""
        for k in keys:
            cells = "".join(f"<td>{m[v][k]:.3f}</td>" for v in m)
            delta = ""
            if "finetuned" in m and "base" in m:
                d = m["finetuned"][k] - m["base"][k]
                delta = f"<td class='{'up' if d > 0 else 'down'}'>{d:+.3f}</td>"
            body += f"<tr><th>{k}</th>{cells}{delta}</tr>"
        dh = "<th>Δ FT vs base</th>" if "finetuned" in m and "base" in m else ""
        ex = ""
        avail = [v for v in VARIANTS if preds[v]]
        for r_i in range(min(10, len(preds[avail[0]]))):
            r0 = preds[avail[0]][r_i]
            cols = "".join(f"<div><h4>{NAMES[v]}</h4><pre>{html.escape(preds[v][r_i]['prediction'])}</pre></div>" for v in avail)
            ex += (f"<details><summary>{html.escape(r0['input'][:140])}</summary>"
                   f"<div class='grid'><div><h4>Reference</h4><pre>{html.escape(r0['reference'])}</pre></div>{cols}</div></details>")
        sections.append(f"<section><h2>{title}</h2><table><tr><th>metric</th>{head}{dh}</tr>{body}</table>"
                        f"<h3>Examples</h3>{ex}</section>")
        if uc == "uc1" and preds["base"] and preds["finetuned"]:
            judge_sample(preds["base"], preds["finetuned"])

    vp, kp = RES / "uc1" / "judge_verdicts.json", RES / "uc1" / "judge_key.json"
    if vp.exists() and kp.exists():  # blind A/B verdicts written before unblinding
        v, k = json.loads(vp.read_text()), json.loads(kp.read_text())
        ft = sum(v[i] == k[i] for i in v); tie = sum(v[i] == "tie" for i in v); bs = len(v) - ft - tie
        allm["uc1_judge"] = {"finetuned_wins": ft, "base_wins": bs, "ties": tie, "ft_win_rate_excl_ties": round(ft / max(1, ft + bs), 3)}
        sections.insert(1, f"<section><h2>UC1 · Blind A/B judge (Claude, n={len(v)})</h2><table><tr><th>Fine-tuned wins</th><td>{ft}</td></tr>"
                           f"<tr><th>Base wins</th><td>{bs}</td></tr><tr><th>Ties</th><td>{tie}</td></tr>"
                           f"<tr><th>FT win-rate (excl. ties)</th><td class='up'>{100 * ft / max(1, ft + bs):.0f}%</td></tr></table>"
                           f"<p>See results/judge.md for rubric and caveats.</p></section>")
    (RES / "metrics.json").write_text(json.dumps(allm, indent=2))
    css = """:root{--bg:#fff;--fg:#1a1a1a;--mut:#f3f4f6;--bd:#d1d5db;--up:#047857;--down:#b91c1c}
@media (prefers-color-scheme:dark){:root{--bg:#111827;--fg:#e5e7eb;--mut:#1f2937;--bd:#374151;--up:#34d399;--down:#f87171}}
body{background:var(--bg);color:var(--fg);font:15px/1.5 system-ui,sans-serif;max-width:1200px;margin:auto;padding:16px}
table{border-collapse:collapse;margin:8px 0}td,th{border:1px solid var(--bd);padding:6px 10px;text-align:right}th:first-child{text-align:left}
.up{color:var(--up);font-weight:600}.down{color:var(--down);font-weight:600}
.grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(240px,1fr));gap:8px}
pre{white-space:pre-wrap;background:var(--mut);padding:8px;border-radius:6px;font-size:13px}
details{border:1px solid var(--bd);border-radius:6px;padding:6px 10px;margin:6px 0}"""
    doc = (f"<!doctype html><html><head><meta charset='utf-8'><meta name='viewport' content='width=device-width,initial-scale=1'>"
           f"<title>Fine-tune Results</title><style>{css}</style></head><body><h1>Qwen2.5-1.5B-Instruct: before vs after LoRA fine-tuning</h1>"
           f"{''.join(sections)}</body></html>")
    (RES / "report.html").write_text(doc, encoding="utf-8")
    print(json.dumps(allm, indent=2))
    print("wrote", RES / "report.html")


if __name__ == "__main__":
    main()
