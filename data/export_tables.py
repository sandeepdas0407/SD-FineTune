"""Export the processed train/val/test splits to CSV (UTF-8 with BOM, opens cleanly in Excel) and one .xlsx per use case.

Usage: python data/export_tables.py   ->  data/export/uc{1,2}_{train,val,test}.csv, uc1_bitext_support.xlsx, uc2_banking77_triage.xlsx
"""
import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "export"
BOOKS = {"uc1": "uc1_bitext_support.xlsx", "uc2": "uc2_banking77_triage.xlsx"}


def load(uc, split):
    rows = [json.loads(l) for l in open(ROOT / "processed" / uc / f"{split}.jsonl", encoding="utf-8")]
    df = pd.DataFrame(rows)
    if uc == "uc1":
        df = df.rename(columns={"input": "customer_message", "target": "reference_reply", "label": "intent"})
        return df[["id", "category", "intent", "customer_message", "reference_reply"]]
    t = df["target"].map(json.loads)  # UC2 target is JSON {"intent", "reply"}
    df["target_reply_synthetic"] = t.map(lambda d: d["reply"])
    df = df.rename(columns={"input": "customer_message", "label": "intent", "target": "target_json"})
    return df[["id", "intent", "customer_message", "target_reply_synthetic", "target_json"]]


def main():
    OUT.mkdir(exist_ok=True)
    for uc, book in BOOKS.items():
        with pd.ExcelWriter(OUT / book, engine="openpyxl") as xw:
            for split in ["train", "val", "test"]:
                df = load(uc, split)
                df.to_csv(OUT / f"{uc}_{split}.csv", index=False, encoding="utf-8-sig")
                df.to_excel(xw, sheet_name=split, index=False)
                ws = xw.sheets[split]
                for col, width in zip("ABCDE", [18, 28, 30, 60, 80] if uc == "uc1" else [16, 34, 60, 70, 80]):
                    ws.column_dimensions[col].width = width
                ws.freeze_panes = "A2"
                print(f"{uc} {split}: {len(df)} rows")
    print("written to", OUT)


if __name__ == "__main__":
    main()
