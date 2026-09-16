#!/usr/bin/env python3
"""
Stage 0 (dataset prep): turn the Kaggle "High-Quality OCR-Ready Invoice PDFs"
batch CSV into the ground-truth eval set used by the rest of the pipeline.

Input:
    data/kaggle_invoices/Batch 1/batch_1.csv
        columns: filename, json_data (ground truth JSON string), ocred_text

Output:
    data/eval_dataset/dataset.json
        list of: {
            "source_file": "invoice_51109301.pdf",
            "input_text": "<ocred_text>",
            "ground_truth": { ...parsed json_data... }
        }

This does NOT run Stage 1 (MinerU) — it uses the dataset's own provided
ocred_text as the model input for now, so we can test the Stage 2 + drift
loop quickly. Swap this for your own Stage 1 output later by pointing
input_text at your extracted markdown instead.
"""

import csv
import json
import sys
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
CSV_PATH = BASE_DIR / "data" / "kaggle_invoices" / "Batch 1" / "batch_1.csv"
OUTPUT_DIR = BASE_DIR / "data" / "eval_dataset"
OUTPUT_FILE = OUTPUT_DIR / "dataset.json"


def main():
    if not CSV_PATH.exists():
        print(f"[error] CSV not found at {CSV_PATH}", file=sys.stderr)
        print("        Update CSV_PATH if you placed the extracted dataset elsewhere.", file=sys.stderr)
        sys.exit(1)

    records = []
    skipped = []

    with open(CSV_PATH, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            filename = row["filename"]
            try:
                ground_truth = json.loads(row["json_data"])
            except json.JSONDecodeError as e:
                skipped.append((filename, str(e)))
                continue

            records.append(
                {
                    "source_file": filename,
                    "input_text": row["ocred_text"],
                    "ground_truth": ground_truth,
                }
            )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_FILE.write_text(json.dumps(records, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"[done] wrote {len(records)} records to {OUTPUT_FILE}")
    if skipped:
        print(f"[warn] skipped {len(skipped)} rows with invalid JSON:", file=sys.stderr)
        for name, err in skipped:
            print(f"  - {name}: {err}", file=sys.stderr)


if __name__ == "__main__":
    main()