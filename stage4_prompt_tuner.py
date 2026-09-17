#!/usr/bin/env python3
"""
Stage 4: self-healing prompt tuning loop.

Reads real mistakes found by Stage 3 (data/mistakes/mistakes.json), builds
a small number of few-shot correction examples from TRAIN-split mistakes
only, appends them to the current production prompt, re-runs Stage 2's
extraction logic against the HOLDOUT split with both the old and new
prompt, and logs everything to MLflow. If the new prompt's holdout
accuracy beats the old prompt's holdout accuracy, it is auto-promoted:
written to prompts/current_system_prompt.txt, which stage2_parse.py reads
on its next run.

This does NOT fine-tune or retrain any model weights. It only versions and
selects prompt TEXT. See stage2_parse.py's DEFAULT_SYSTEM_PROMPT for the
baseline this improves on.

Requires a local MLflow tracking server running separately, e.g.:
    mlflow server --backend-store-uri sqlite:///mlflow.db --default-artifact-root ./mlruns

Usage:
    python stage4_prompt_tuner.py
    python stage4_prompt_tuner.py --tracking-uri http://127.0.0.1:5000
    python stage4_prompt_tuner.py --num-examples 3 --device cuda
"""

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any, Optional

import mlflow

BASE_DIR = Path(__file__).resolve().parent
MISTAKES_FILE = BASE_DIR / "data" / "mistakes" / "mistakes.json"
EVAL_DATASET_FILE = BASE_DIR / "data" / "eval_dataset" / "dataset.json"
PROMPT_FILE = BASE_DIR / "prompts" / "current_system_prompt.txt"
MODEL_DIR = BASE_DIR / "models" / "qwen2.5-1.5b"
MODEL_ID = "Qwen/Qwen2.5-1.5B-Instruct"

EXPERIMENT_NAME = "invoice_prompt_self_healing"
MAX_INPUT_CHARS = 6000

# Import the exact same default prompt, extraction, and chain-building logic
# Stage 2 uses, so Stage 4 evaluates prompts identically to how they'll
# actually run in production.
sys.path.insert(0, str(BASE_DIR))
from stage2_parse import (  # noqa: E402
    DEFAULT_SYSTEM_PROMPT,
    build_chain,
    ensure_model_downloaded,
    extract_json,
)


# ---- Loading current production prompt -------------------------------------


def load_current_prompt() -> str:
    if PROMPT_FILE.exists():
        text = PROMPT_FILE.read_text(encoding="utf-8").strip()
        if text:
            return text
    return DEFAULT_SYSTEM_PROMPT


# ---- Building few-shot correction examples from real mistakes --------------


def select_exemplars(mistakes: list, num_examples: int, priority_fields: Optional[list] = None) -> list:
    """Pick TRAIN-split mistakes to turn into few-shot correction examples.

    If priority_fields is given (e.g. ["client.address", "seller.address"]),
    records containing errors in those fields are ranked first — this lets
    the loop target whatever error pattern Stage 3 just reported as most
    common, rather than picking exemplars that happen to have the most
    total diffs but miss the actual current bottleneck. Falls back to
    diff-count ranking within each group, and skips records with no usable
    input_text or ground_truth."""
    candidates = [
        m for m in mistakes
        if m.get("split") == "train" and m.get("input_text") and m.get("ground_truth")
    ]

    if priority_fields:
        def matches_priority(m):
            diff_fields = {d["field"] for d in m.get("diffs", [])}
            return any(
                any(field.startswith(pf) for pf in priority_fields)
                for field in diff_fields
            )

        priority_candidates = [m for m in candidates if matches_priority(m)]
        other_candidates = [m for m in candidates if not matches_priority(m)]
        priority_candidates.sort(key=lambda m: len(m.get("diffs", [])), reverse=True)
        other_candidates.sort(key=lambda m: len(m.get("diffs", [])), reverse=True)
        candidates = priority_candidates + other_candidates
    else:
        candidates.sort(key=lambda m: len(m.get("diffs", [])), reverse=True)

    return candidates[:num_examples]


def build_updated_prompt(base_prompt: str, exemplars: list) -> str:
    if not exemplars:
        return base_prompt

    prompt = base_prompt + "\n\n### CORRECTIONS FROM REAL MISTAKES:\n"
    prompt += (
        "The examples below show real cases where a previous extraction was "
        "WRONG, and the CORRECT extraction that should have been produced. "
        "Learn the pattern of each correction and apply it generally.\n"
    )
    # Cap input text per exemplar more aggressively than a normal document:
    # each exemplar's OCR text AND ground truth JSON both get added to every
    # future prompt, so with multiple exemplars this compounds fast and can
    # push the combined system prompt + per-document input past the model's
    # context window, causing the pipeline to error on EVERY call silently
    # (see run_extraction's debug logging if this happens again).
    EXEMPLAR_TEXT_CHARS = 1500
    for idx, ex in enumerate(exemplars, 1):
        text = ex["input_text"]
        if len(text) > EXEMPLAR_TEXT_CHARS:
            text = text[:EXEMPLAR_TEXT_CHARS] + " [...truncated]"
        prompt += f"\n--- Correction example {idx} ---\n"
        prompt += f"Input Document Text:\n{text}\n"
        prompt += f"CORRECT Target JSON:\n{json.dumps(ex['ground_truth'], ensure_ascii=False)}\n"
    return prompt


# ---- Evaluating a prompt against the holdout split --------------------------


def run_extraction(chain, text: str, debug_state: dict, label: str = "") -> Optional[dict]:
    if len(text) > MAX_INPUT_CHARS:
        text = text[:MAX_INPUT_CHARS]
    try:
        raw_output = chain.invoke({"document": text})
        return extract_json(raw_output)
    except Exception as e:
        if debug_state["count"] < 3:
            debug_state["count"] += 1
            print(f"    [debug:{label}] extraction failed ({e}); raw output preview:")
            try:
                print(f"    [debug:{label}] {raw_output[:500]!r}")
            except NameError:
                print(f"    [debug:{label}] (chain.invoke itself raised, no raw_output captured)")
        return None


def flatten(obj: Any, prefix: str = "") -> dict:
    flat = {}
    if isinstance(obj, dict):
        for k, v in obj.items():
            path = f"{prefix}.{k}" if prefix else k
            flat.update(flatten(v, path))
    elif isinstance(obj, list):
        for i, v in enumerate(obj):
            flat.update(flatten(v, f"{prefix}[{i}]"))
    else:
        flat[prefix] = obj
    return flat


def is_exact_match(prediction: Optional[dict], ground_truth: dict) -> bool:
    if prediction is None:
        return False
    flat_pred = flatten(prediction)
    flat_gt = flatten(ground_truth)
    all_fields = set(flat_pred.keys()) | set(flat_gt.keys())
    for field in all_fields:
        if str(flat_pred.get(field, "<missing>")).strip() != str(flat_gt.get(field, "<missing>")).strip():
            return False
    return True


def field_level_accuracy(prediction: Optional[dict], ground_truth: dict) -> tuple:
    """Return (correct_fields, total_fields) comparing every flattened leaf
    field. Unlike exact-match (all-or-nothing per document), this credits
    partial correctness — useful when documents have 20+ fields and getting
    19/20 right is real progress that exact-match would score identically
    to getting 0/20 right."""
    flat_gt = flatten(ground_truth)
    total = len(flat_gt)
    if prediction is None:
        return 0, total
    flat_pred = flatten(prediction)
    correct = sum(
        1
        for field, expected in flat_gt.items()
        if str(flat_pred.get(field, "<missing>")).strip() == str(expected).strip()
    )
    return correct, total


def evaluate_prompt(chain, holdout_records: list, label: str = "") -> dict:
    exact_correct = 0
    total_docs = len(holdout_records)
    field_correct_sum = 0
    field_total_sum = 0
    debug_state = {"count": 0}

    for i, r in enumerate(holdout_records, 1):
        pred = run_extraction(chain, r["input_text"], debug_state, label=label)
        exact = is_exact_match(pred, r["ground_truth"])
        f_correct, f_total = field_level_accuracy(pred, r["ground_truth"])

        if exact:
            exact_correct += 1
        field_correct_sum += f_correct
        field_total_sum += f_total

        field_pct = f_correct / f_total if f_total else 0.0
        status = "EXACT MATCH" if exact else f"{f_correct}/{f_total} fields ({field_pct:.0%})"
        print(f"  [{i}/{total_docs}] {r['source_file']}: {status}")

    return {
        "exact_correct": exact_correct,
        "exact_total": total_docs,
        "exact_accuracy": round(exact_correct / total_docs, 4) if total_docs else 0.0,
        "field_correct": field_correct_sum,
        "field_total": field_total_sum,
        "field_accuracy": round(field_correct_sum / field_total_sum, 4) if field_total_sum else 0.0,
    }


# ---- Main loop ---------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(description="Self-healing prompt tuning loop")
    parser.add_argument("--tracking-uri", default="http://127.0.0.1:5000", help="MLflow tracking server URI")
    parser.add_argument("--num-examples", type=int, default=3, help="Number of few-shot correction examples to add")
    parser.add_argument("--device", default="auto", choices=["auto", "cuda", "cpu"])
    args = parser.parse_args()

    device = args.device
    if device == "auto":
        try:
            import torch
            device = "cuda" if torch.cuda.is_available() else "cpu"
        except ImportError:
            device = "cpu"
    print(f"[device] Using device: {device}")

    mlflow.set_tracking_uri(args.tracking_uri)
    mlflow.set_experiment(EXPERIMENT_NAME)

    if not MISTAKES_FILE.exists():
        print(f"[error] {MISTAKES_FILE} not found. Run stage2_parse.py + stage3_find_mistakes.py first.", file=sys.stderr)
        sys.exit(1)
    if not EVAL_DATASET_FILE.exists():
        print(f"[error] {EVAL_DATASET_FILE} not found. Run stage0_build_eval_dataset.py first.", file=sys.stderr)
        sys.exit(1)

    mistakes = json.loads(MISTAKES_FILE.read_text(encoding="utf-8"))
    eval_records = json.loads(EVAL_DATASET_FILE.read_text(encoding="utf-8"))
    holdout_records = [r for r in eval_records if r.get("split") == "holdout"]

    if not holdout_records:
        print("[error] No holdout records found. Re-run stage0_build_eval_dataset.py (updated version).", file=sys.stderr)
        sys.exit(1)

    current_prompt = load_current_prompt()

    # Auto-derive which field(s) to prioritize: whatever base field name
    # (before any [index]) shows up most often across all train-split
    # mistakes right now. This keeps each loop iteration targeting whatever
    # the CURRENT dominant error is, without needing to hardcode it.
    base_field_counts: dict = {}
    for m in mistakes:
        if m.get("split") != "train":
            continue
        for d in m.get("diffs", []):
            base_field = re.sub(r"\[\d+\]", "", d["field"])  # items[0].description -> items.description
            base_field_counts[base_field] = base_field_counts.get(base_field, 0) + 1
    top_fields = sorted(base_field_counts, key=base_field_counts.get, reverse=True)[:3]
    if top_fields:
        print(f"[info] Current top error field(s) on train split: {top_fields}")

    exemplars = select_exemplars(mistakes, args.num_examples, priority_fields=top_fields)

    if not exemplars:
        print("[info] No usable train-split mistakes found to build corrections from. Nothing to do.")
        sys.exit(0)

    print(f"[info] Building new prompt from {len(exemplars)} train-split mistake(s):")
    for ex in exemplars:
        print(f"  - {ex['source_file']} ({len(ex.get('diffs', []))} field diffs)")

    new_prompt = build_updated_prompt(current_prompt, exemplars)

    model_path = ensure_model_downloaded()

    # Fail fast and loudly if the prompt is too long for the model's context
    # window, rather than silently burning ~20 minutes generating garbage
    # for every holdout document (each one would raise inside
    # run_extraction and get recorded as a 0% result, which looks like a
    # bad prompt rather than an infrastructure problem).
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(str(model_path))
    new_prompt_tokens = len(tokenizer.encode(new_prompt))
    max_context = getattr(tokenizer, "model_max_length", 32768)
    # Reserve room for the per-document input text and the generation budget
    RESERVED_FOR_INPUT_AND_OUTPUT = 2048 + 2048
    if max_context and max_context < 1_000_000 and new_prompt_tokens + RESERVED_FOR_INPUT_AND_OUTPUT > max_context:
        print(
            f"[error] New prompt is {new_prompt_tokens} tokens, which combined with "
            f"reserved space for input+output ({RESERVED_FOR_INPUT_AND_OUTPUT}) exceeds "
            f"the model's context window ({max_context}). Reduce --num-examples or "
            f"shorten exemplar text before continuing.",
            file=sys.stderr,
        )
        sys.exit(1)
    print(f"[info] New prompt size: {new_prompt_tokens} tokens (context window: {max_context})")

    print("[eval] Evaluating CURRENT (production) prompt on holdout set...")
    old_chain = build_chain(model_path, device, current_prompt)
    old_results = evaluate_prompt(old_chain, holdout_records, label="OLD")
    print(
        f"[eval] Current prompt — field accuracy: {old_results['field_accuracy']:.1%} "
        f"({old_results['field_correct']}/{old_results['field_total']}), "
        f"exact-match: {old_results['exact_accuracy']:.1%} ({old_results['exact_correct']}/{old_results['exact_total']})"
    )

    # Free the first model's GPU memory before loading a second instance.
    # Without this, both models sit in VRAM simultaneously — two 1.5B-param
    # instances (weights + KV cache + activations) don't fit on a 6GB card,
    # so the second evaluation pass OOMs on every single document, which
    # silently produces a flat 0% result that looks like a bad prompt
    # rather than the infrastructure problem it actually is.
    try:
        # Reach into the chain (prompt | ChatHuggingFace(llm=HuggingFacePipeline) | parser)
        # to find and explicitly delete the underlying transformers model.
        # `del old_chain` alone may not drop the refcount to zero since
        # LangChain wraps the pipeline in several layers.
        hf_chat_model = old_chain.middle[0] if hasattr(old_chain, "middle") else None
        if hf_chat_model is not None:
            underlying_pipeline = getattr(hf_chat_model.llm, "pipeline", None)
            if underlying_pipeline is not None:
                del underlying_pipeline.model
    except Exception as e:
        print(f"[warn] Could not explicitly free model reference: {e}")
    del old_chain
    if device == "cuda":
        import torch
        import gc
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        print(f"[info] Freed old model from GPU. Current allocated: {torch.cuda.memory_allocated() / 1e9:.2f} GB")

    print("[eval] Evaluating NEW (candidate) prompt on holdout set...")
    new_chain = build_chain(model_path, device, new_prompt)
    new_results = evaluate_prompt(new_chain, holdout_records, label="NEW")
    print(
        f"[eval] New prompt — field accuracy: {new_results['field_accuracy']:.1%} "
        f"({new_results['field_correct']}/{new_results['field_total']}), "
        f"exact-match: {new_results['exact_accuracy']:.1%} ({new_results['exact_correct']}/{new_results['exact_total']})"
    )

    # Field-level accuracy is the primary promotion criterion: with a schema
    # this granular (20+ leaf fields per document), exact-match can stay at
    # 0% even when a prompt is making real, measurable progress. Require a
    # minimum improvement margin (not just >) so noise from generation
    # nondeterminism-adjacent effects (long outputs, borderline formatting)
    # doesn't trigger promotion on a coin-flip-sized difference.
    MIN_IMPROVEMENT = 0.01  # 1 percentage point
    improved = (new_results["field_accuracy"] - old_results["field_accuracy"]) >= MIN_IMPROVEMENT

    with mlflow.start_run(run_name=f"prompt_update_{len(exemplars)}ex") as run:
        mlflow.log_param("num_exemplars", len(exemplars))
        mlflow.log_param("exemplar_files", [ex["source_file"] for ex in exemplars])
        mlflow.log_param("holdout_size", len(holdout_records))
        mlflow.log_param("device", device)
        mlflow.log_param("min_improvement_threshold", MIN_IMPROVEMENT)

        mlflow.log_metric("holdout_field_accuracy_old", old_results["field_accuracy"])
        mlflow.log_metric("holdout_field_accuracy_new", new_results["field_accuracy"])
        mlflow.log_metric("holdout_exact_accuracy_old", old_results["exact_accuracy"])
        mlflow.log_metric("holdout_exact_accuracy_new", new_results["exact_accuracy"])
        mlflow.log_metric("holdout_field_correct_old", old_results["field_correct"])
        mlflow.log_metric("holdout_field_correct_new", new_results["field_correct"])

        mlflow.log_text(current_prompt, artifact_file="old_system_prompt.txt")
        mlflow.log_text(new_prompt, artifact_file="new_system_prompt.txt")
        mlflow.log_dict(exemplars, artifact_file="exemplars.json")
        mlflow.log_dict(
            {"old": old_results, "new": new_results, "promoted": improved},
            artifact_file="evaluation_result.json",
        )

        mlflow.set_tag("promoted", str(improved))

        if improved:
            PROMPT_FILE.parent.mkdir(parents=True, exist_ok=True)
            PROMPT_FILE.write_text(new_prompt, encoding="utf-8")
            print(
                f"[promote] New prompt IMPROVED holdout field accuracy "
                f"({old_results['field_accuracy']:.1%} -> {new_results['field_accuracy']:.1%}). Promoted to production."
            )
            mlflow.log_param("promoted_to", str(PROMPT_FILE))
        else:
            print(
                f"[skip] New prompt did NOT improve holdout field accuracy by at least {MIN_IMPROVEMENT:.0%} "
                f"({old_results['field_accuracy']:.1%} -> {new_results['field_accuracy']:.1%}). Not promoted."
            )

        print(f"[done] Run logged: {run.info.run_id} (experiment: {EXPERIMENT_NAME})")


if __name__ == "__main__":
    main()