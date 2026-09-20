#!/usr/bin/env python3
"""
Streamlit frontend for the self-healing invoice extraction pipeline.

Run with:
    streamlit run app.py

Requires the other pipeline files (stage0_build_eval_dataset.py,
stage2_parse.py, stage3_find_mistakes.py, stage4_prompt_tuner.py) to be
in the same directory, and an MLflow server running at
http://127.0.0.1:5000 for the "Prompt History" tab to work.
"""

import json
import subprocess
import sys
import time
from pathlib import Path

import streamlit as st

BASE_DIR = Path(__file__).resolve().parent
EVAL_DATASET_FILE = BASE_DIR / "data" / "eval_dataset" / "dataset.json"
PREDICTIONS_FILE = BASE_DIR / "data" / "structured_json" / "predictions.json"
MISTAKES_FILE = BASE_DIR / "data" / "mistakes" / "mistakes.json"
SUMMARY_FILE = BASE_DIR / "data" / "mistakes" / "summary.json"
PROMPT_FILE = BASE_DIR / "prompts" / "current_system_prompt.txt"
MLFLOW_TRACKING_URI = "http://127.0.0.1:5000"
MLFLOW_EXPERIMENT = "invoice_prompt_self_healing"

st.set_page_config(page_title="Invoice Extraction Pipeline", layout="wide")


# ---- Cached, expensive resources -------------------------------------------


@st.cache_resource(show_spinner="Loading Qwen2.5-1.5B model (first time only)...")
def get_extraction_chain(device: str):
    """Build the Stage 2 extraction chain once per Streamlit session and
    device choice. Cached so re-running extraction on new uploads doesn't
    reload the model every time."""
    sys.path.insert(0, str(BASE_DIR))
    from stage2_parse import build_chain, ensure_model_downloaded, load_system_prompt

    model_path = ensure_model_downloaded()
    system_prompt = load_system_prompt()
    chain = build_chain(model_path, device, system_prompt)
    return chain


def resolve_device(choice: str) -> str:
    if choice == "auto":
        try:
            import torch
            return "cuda" if torch.cuda.is_available() else "cpu"
        except ImportError:
            return "cpu"
    return choice


def extract_pdf_text(uploaded_file) -> str:
    """Plain text extraction via PyMuPDF — NOT the full MinerU OCR model
    used by stage1_ingest.py. Good enough for text-based PDFs; for scanned
    images, use stage1_ingest.py separately first."""
    import fitz

    pdf_bytes = uploaded_file.read()
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    text = "\n".join(page.get_text() for page in doc)
    doc.close()
    return text


def load_json_safe(path: Path):
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return None


def run_stage_subprocess(script_name: str, args: list, log_placeholder):
    """Run a pipeline stage as a subprocess, streaming its output live into
    a Streamlit placeholder. Runs as a subprocess (not in-process) so a
    long GPU job doesn't block or crash the Streamlit server itself."""
    cmd = [sys.executable, str(BASE_DIR / script_name)] + args
    process = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
        cwd=str(BASE_DIR),
    )
    log_lines = []
    for line in process.stdout:
        log_lines.append(line.rstrip())
        log_placeholder.code("\n".join(log_lines[-40:]), language="text")
    process.wait()
    return process.returncode, "\n".join(log_lines)


# ---- Sidebar: global device choice -----------------------------------------

st.sidebar.title("Settings")
device_choice = st.sidebar.selectbox("Device", ["auto", "cuda", "cpu"], index=0)
st.sidebar.caption(
    "Used for both live extraction (this app) and any pipeline stage "
    "triggered from the Pipeline tab."
)

tab_extract, tab_pipeline, tab_results, tab_prompt = st.tabs(
    ["Extract Invoice", "Run Pipeline", "Mistakes & Accuracy", "Prompt History"]
)


# ---- Tab 1: single-invoice extraction ---------------------------------------

with tab_extract:
    st.header("Extract a single invoice")
    st.caption(
        "Runs the current production prompt (prompts/current_system_prompt.txt, "
        "or the built-in default if none exists yet) live against your input."
    )

    input_mode = st.radio("Input type", ["Paste OCR text", "Upload PDF"], horizontal=True)

    document_text = ""
    if input_mode == "Paste OCR text":
        document_text = st.text_area("Invoice OCR text", height=250)
    else:
        uploaded = st.file_uploader("Upload a PDF", type=["pdf"])
        if uploaded is not None:
            document_text = extract_pdf_text(uploaded)
            st.caption(
                "Extracted with a plain-text PDF reader, not the full MinerU OCR "
                "model. For scanned/image-based PDFs, run stage1_ingest.py "
                "separately first and paste its output instead."
            )
            with st.expander("Show extracted text"):
                st.text(document_text)

    if st.button("Run extraction", type="primary", disabled=not document_text.strip()):
        device = resolve_device(device_choice)
        with st.spinner(f"Running extraction on {device}..."):
            try:
                chain = get_extraction_chain(device)
                from stage2_parse import extract_json

                raw_output = chain.invoke({"document": document_text})
                try:
                    result = extract_json(raw_output)
                    st.success("Extraction succeeded")
                    st.json(result)
                except ValueError as e:
                    st.error(f"Model output could not be parsed as JSON: {e}")
                    st.text_area("Raw model output", raw_output, height=200)
            except Exception as e:
                st.error(f"Extraction failed: {e}")


# ---- Tab 2: pipeline runner --------------------------------------------------

with tab_pipeline:
    st.header("Run pipeline stages")
    st.caption(
        "Each stage runs as a separate process with live log output below. "
        "Run in order for a full loop: Stage 0 -> Stage 2 -> Stage 3 -> Stage 4."
    )

    col0, col2, col3, col4 = st.columns(4)

    with col0:
        st.subheader("Stage 0")
        st.caption("Build eval dataset + train/holdout split")
        if st.button("Run Stage 0"):
            log_area = st.empty()
            code, _ = run_stage_subprocess("stage0_build_eval_dataset.py", [], log_area)
            (st.success if code == 0 else st.error)(f"Stage 0 exited with code {code}")

    with col2:
        st.subheader("Stage 2")
        st.caption("Parse all invoices with current prompt")
        limit = st.number_input("Limit (0 = all)", min_value=0, value=0, step=1, key="limit2")
        if st.button("Run Stage 2"):
            device = resolve_device(device_choice)
            args = ["--device", device]
            if limit > 0:
                args += ["--limit", str(limit)]
            log_area = st.empty()
            code, _ = run_stage_subprocess("stage2_parse.py", args, log_area)
            (st.success if code == 0 else st.error)(f"Stage 2 exited with code {code}")

    with col3:
        st.subheader("Stage 3")
        st.caption("Find mistakes vs ground truth")
        if st.button("Run Stage 3"):
            log_area = st.empty()
            code, _ = run_stage_subprocess("stage3_find_mistakes.py", [], log_area)
            (st.success if code == 0 else st.error)(f"Stage 3 exited with code {code}")

    with col4:
        st.subheader("Stage 4")
        st.caption("Self-healing prompt tuning loop")
        num_examples = st.number_input("Few-shot examples", min_value=1, value=3, step=1, key="numex4")
        if st.button("Run Stage 4"):
            device = resolve_device(device_choice)
            args = ["--device", device, "--num-examples", str(num_examples), "--tracking-uri", MLFLOW_TRACKING_URI]
            log_area = st.empty()
            st.warning("This runs two full holdout passes and can take a long time on a small GPU.")
            code, _ = run_stage_subprocess("stage4_prompt_tuner.py", args, log_area)
            (st.success if code == 0 else st.error)(f"Stage 4 exited with code {code}")

    st.divider()
    st.caption(
        "MLflow server must be running separately: `mlflow server --backend-store-uri "
        "sqlite:///mlflow.db --default-artifact-root ./mlruns --host 127.0.0.1 --port 5000`"
    )


# ---- Tab 3: mistakes & accuracy browser -------------------------------------

with tab_results:
    st.header("Latest run: mistakes and accuracy")

    summary = load_json_safe(SUMMARY_FILE)
    if summary is None:
        st.info("No summary.json found yet. Run Stage 2 and Stage 3 first.")
    else:
        col1, col2, col3 = st.columns(3)
        col1.metric("Total records", summary["total_records"])
        col2.metric("Exact match (overall)", f"{summary['accuracy_exact_match']:.1%}")
        col3.metric("Parse failures", summary["parse_failures"])

        by_split = summary.get("accuracy_by_split", {})
        if by_split:
            col_a, col_b = st.columns(2)
            train_acc = by_split.get("train", {}).get("accuracy")
            holdout_acc = by_split.get("holdout", {}).get("accuracy")
            if train_acc is not None:
                col_a.metric("Train accuracy", f"{train_acc:.1%}")
            if holdout_acc is not None:
                col_b.metric("Holdout accuracy", f"{holdout_acc:.1%}")

        st.subheader("Most common field errors")
        field_errors = summary.get("most_common_field_errors", [])
        if field_errors:
            st.bar_chart({field: count for field, count in field_errors})

        st.subheader("Per-document mistakes")
        mistakes = load_json_safe(MISTAKES_FILE) or []
        if mistakes:
            filenames = [m["source_file"] for m in mistakes]
            selected = st.selectbox("Select a document to inspect", filenames)
            record = next(m for m in mistakes if m["source_file"] == selected)

            st.caption(f"Split: {record.get('split', 'unknown')}")
            if record.get("parse_error"):
                st.error(f"Parse error: {record['parse_error']}")

            diff_col, gt_col = st.columns(2)
            with diff_col:
                st.write("**Field diffs**")
                st.dataframe(record.get("diffs", []), use_container_width=True)
            with gt_col:
                st.write("**Ground truth**")
                st.json(record.get("ground_truth", {}))
        else:
            st.info("No mistakes recorded (or mistakes.json not found).")


# ---- Tab 4: prompt history via MLflow ---------------------------------------

with tab_prompt:
    st.header("Prompt version history")

    current_prompt = None
    if PROMPT_FILE.exists():
        current_prompt = PROMPT_FILE.read_text(encoding="utf-8")
        st.subheader("Current production prompt")
        with st.expander("Show current prompt", expanded=False):
            st.text(current_prompt)
    else:
        st.info("No promoted prompt yet — Stage 2 is using its built-in default.")

    st.subheader("MLflow run history")
    try:
        import mlflow

        mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
        client = mlflow.tracking.MlflowClient()
        experiment = client.get_experiment_by_name(MLFLOW_EXPERIMENT)

        if experiment is None:
            st.info(f"No MLflow experiment named '{MLFLOW_EXPERIMENT}' yet. Run Stage 4 at least once.")
        else:
            runs = client.search_runs([experiment.experiment_id], order_by=["start_time DESC"])
            if not runs:
                st.info("No runs logged yet.")
            else:
                rows = []
                for r in runs:
                    m = r.data.metrics
                    p = r.data.params
                    tags = r.data.tags
                    rows.append(
                        {
                            "run_id": r.info.run_id[:8],
                            "started": time.strftime("%Y-%m-%d %H:%M", time.localtime(r.info.start_time / 1000)),
                            "num_exemplars": p.get("num_exemplars"),
                            "field_acc_old": m.get("holdout_field_accuracy_old"),
                            "field_acc_new": m.get("holdout_field_accuracy_new"),
                            "exact_acc_old": m.get("holdout_exact_accuracy_old"),
                            "exact_acc_new": m.get("holdout_exact_accuracy_new"),
                            "promoted": tags.get("promoted"),
                        }
                    )
                st.dataframe(rows, use_container_width=True)

                st.caption(f"Full details for any run: {MLFLOW_TRACKING_URI}/#/experiments/{experiment.experiment_id}")
    except Exception as e:
        st.warning(f"Could not reach MLflow at {MLFLOW_TRACKING_URI}: {e}")
        st.caption("Make sure the MLflow server is running (see Run Pipeline tab).")