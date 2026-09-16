#!/usr/bin/env python3
"""
Stage 3: compare Stage 2 predictions against ground truth and collect
"mistakes" — the real drift signal that Stage 4's self-healing prompt loop
will later consume.

A mistake is recorded per source_file when any of:
  - the model's output failed to parse as JSON at all
  - a top-level or nested field differs from ground truth
  - a field present in ground truth is missing from the prediction

Input:
    data/structured_json/predictions.json  (from stage2_parse.py)

Output:
    data/mistakes/mistakes.json   -> full list of per-document diffs
    data/mistakes/summary.json    -> aggregate stats (field-level error rates)

This uses exact string comparison, matching the ground truth's convention
of keeping numbers as formatted strings. If your prompt/schema conventions
drift from that, the diff will over-report mismatches — check summary.json's
most_common_field_errors first if error rates look suspiciously high.
"""

import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

BASE_DIR = Path(__file__).resolve().parent
PREDICTIONS_FILE = BASE_DIR / "data" / "structured_json" / "predictions.json"
OUTPUT_DIR = BASE_DIR / "data" / "mistakes"
MISTAKES_FILE = OUTPUT_DIR / "mistakes.json"
SUMMARY_FILE = OUTPUT_DIR / "summary.json"


def flatten(obj: Any, prefix: str = "") -> dict:
    """Flatten nested dicts/lists into dotted-path -> value pairs for diffing.
    e.g. {"seller": {"name": "X"}} -> {"seller.name": "X"}
         {"items": [{"description": "A"}]} -> {"items[0].description": "A"}
    """
    flat = {}
    if isinstance(obj, dict):
        for k, v in obj.items():
            path = f"{prefix}.{k}" if prefix else k
            flat.update(flatten(v, path))
    elif isinstance(obj, list):
        for i, v in enumerate(obj):
            path = f"{prefix}[{i}]"
            flat.update(flatten(v, path))
    else:
        flat[prefix] = obj
    return flat


def diff_record(prediction: Any, ground_truth: Any) -> list:
    """Return a list of {field, expected, actual} for every mismatch."""
    if prediction is None:
        return [{"field": "<entire document>", "expected": "<valid JSON>", "actual": "<parse failure>"}]

    flat_pred = flatten(prediction)
    flat_gt = flatten(ground_truth)

    mismatches = []
    all_fields = set(flat_pred.keys()) | set(flat_gt.keys())
    for field in sorted(all_fields):
        expected = flat_gt.get(field, "<missing in ground truth>")
        actual = flat_pred.get(field, "<missing in prediction>")
        if str(expected).strip() != str(actual).strip():
            mismatches.append({"field": field, "expected": expected, "actual": actual})

    return mismatches


def main():
    if not PREDICTIONS_FILE.exists():
        print(f"[error] {PREDICTIONS_FILE} not found. Run stage2_parse.py first.", file=sys.stderr)
        sys.exit(1)

    predictions = json.loads(PREDICTIONS_FILE.read_text(encoding="utf-8"))

    mistakes = []
    field_error_counter = Counter()
    parse_failures = 0
    clean_records = 0

    for record in predictions:
        source_file = record["source_file"]
        prediction = record.get("prediction")
        ground_truth = record.get("ground_truth")
        parse_error = record.get("parse_error")

        if parse_error:
            parse_failures += 1

        diffs = diff_record(prediction, ground_truth)

        if diffs:
            for d in diffs:
                field_error_counter[d["field"]] += 1
            mistakes.append(
                {
                    "source_file": source_file,
                    "input_text": None,  # filled in below if available
                    "prediction": prediction,
                    "ground_truth": ground_truth,
                    "raw_model_output": record.get("raw_model_output"),
                    "parse_error": parse_error,
                    "diffs": diffs,
                }
            )
        else:
            clean_records += 1

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    MISTAKES_FILE.write_text(json.dumps(mistakes, indent=2, ensure_ascii=False), encoding="utf-8")

    total = len(predictions)
    summary = {
        "total_records": total,
        "clean_records": clean_records,
        "records_with_mistakes": len(mistakes),
        "parse_failures": parse_failures,
        "accuracy_exact_match": round(clean_records / total, 4) if total else 0.0,
        "most_common_field_errors": field_error_counter.most_common(20),
    }
    SUMMARY_FILE.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"[done] {clean_records}/{total} records matched ground truth exactly")
    print(f"[done] {len(mistakes)} records had at least one mistake ({parse_failures} outright parse failures)")
    print(f"[done] wrote {MISTAKES_FILE}")
    print(f"[done] wrote {SUMMARY_FILE}")
    print()
    print("Top field-level error locations:")
    for field, count in field_error_counter.most_common(10):
        print(f"  {count:4d}x  {field}")


if __name__ == "__main__":
    main()