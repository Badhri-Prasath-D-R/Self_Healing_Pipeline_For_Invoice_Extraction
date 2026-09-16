#!/usr/bin/env python3
"""
Stage 2: parse extracted invoice text into structured JSON matching the
FIXED schema used by our eval dataset (Kaggle "High-Quality OCR-Ready
Invoice PDFs"):

{
  "invoice_no": str,
  "date_of_issue": str,          # as written, e.g. "03/07/2023"
  "seller": {
    "name": str, "address": str, "tax_id": str, "gstin": str (optional)
  },
  "client": {
    "name": str, "address": str, "tax_id": str
  },
  "items": [
    {
      "item_no": str, "description": str, "quantity": str,
      "unit_measure": str, "net_price": str, "net_worth": str,
      "vat_pct": str, "gross_worth": str
    }
  ],
  "summary": {
    "vat_pct": str, "net_worth": str, "vat_amount": str,
    "gross_worth": str, "total_net_worth": str, "total_vat": str,
    "total_gross_worth": str
  }
}

NOTE: in the ground-truth dataset all numeric-looking values (prices,
quantities, totals) are kept as STRINGS, formatted with thousands
commas (e.g. "74,120.00") and totals are prefixed with currency
(e.g. "INR 1,676,976.00"). The prompt below asks the model to mirror
that convention so its output can be diffed directly against ground
truth without a normalization step. If your real-world documents don't
follow this convention, loosen this prompt later.

Directory layout:
    data/eval_dataset/dataset.json  -> input (from stage0_build_eval_dataset.py)
    data/structured_json/           -> output predictions
    models/qwen2.5-1.5b/            -> local cache for the downloaded LLM

Requirements:
    pip install langchain langchain-huggingface transformers torch huggingface_hub
"""

import json
import re
import sys
from pathlib import Path
from typing import Any

MODEL_ID = "Qwen/Qwen2.5-1.5B-Instruct"

BASE_DIR = Path(__file__).resolve().parent
EVAL_DATASET_FILE = BASE_DIR / "data" / "eval_dataset" / "dataset.json"
OUTPUT_DIR = BASE_DIR / "data" / "structured_json"
OUTPUT_FILE = OUTPUT_DIR / "predictions.json"
MODEL_DIR = BASE_DIR / "models" / "qwen2.5-1.5b"
PROMPT_FILE = BASE_DIR / "prompts" / "current_system_prompt.txt"

MAX_INPUT_CHARS = 6000

DEFAULT_SYSTEM_PROMPT = """You are a strict invoice information-extraction engine.

Read the OCR text of an invoice and extract fields into EXACTLY this JSON \
schema — use these exact key names, this exact nesting, every time:

{
  "invoice_no": "<string>",
  "date_of_issue": "<string, as written in the document, e.g. 03/07/2023>",
  "seller": {
    "name": "<string>",
    "address": "<string>",
    "tax_id": "<string>",
    "gstin": "<string, omit key if not present>"
  },
  "client": {
    "name": "<string>",
    "address": "<string>",
    "tax_id": "<string>"
  },
  "items": [
    {
      "item_no": "<string, e.g. '1.'>",
      "description": "<string>",
      "quantity": "<string, e.g. '9.00'>",
      "unit_measure": "<string, e.g. 'pcs'>",
      "net_price": "<string, formatted with thousands commas, e.g. '74,120.00'>",
      "net_worth": "<string, same number formatting>",
      "vat_pct": "<string, e.g. '10%'>",
      "gross_worth": "<string, same number formatting>"
    }
  ],
  "summary": {
    "vat_pct": "<string>",
    "net_worth": "<string>",
    "vat_amount": "<string>",
    "gross_worth": "<string>",
    "total_net_worth": "<string, prefixed with currency, e.g. 'INR 1,676,976.00'>",
    "total_vat": "<string, prefixed with currency>",
    "total_gross_worth": "<string, prefixed with currency>"
  }
}

Rules:
- Keep ALL numeric-looking values (prices, quantities, totals) as STRINGS, \
formatted with thousands-separator commas and two decimal places, exactly \
as they would appear printed on the invoice (e.g. "74,120.00", not 74120).
- The three "total_*" fields in summary should be prefixed with the \
currency code exactly as printed (e.g. "INR 1,676,976.00").
- If the document has only one item, "items" is still a list with one entry.
- List EVERY line item individually, in full, even if there are many or \
they look repetitive. NEVER shorten the list with a placeholder like \
"// ... other items" or "etc." — that produces invalid JSON and loses data.
- If a field genuinely cannot be found, use an empty string "" rather than \
inventing a value.
- Return ONLY the single JSON object. No prose, no markdown code fences, \
no explanations, no trailing commentary.

IMPORTANT — seller/client block layout:
The OCR text often lays the seller and client out as two side-by-side \
columns that get flattened into merged lines, like this:

  Seller: Client:
  TechVision Distributors Pvt Ltd Raj Electronics Pvt Ltd
  Plot 14, MIDC Industrial Area, Andheri East 42 MG Road
  Mumbai, Maharashtra - 400093 Bengaluru, Karnataka - 560001
  Tax Id: 27AABCT1234F1Z5 Tax Id: 901-95-4704
  GSTIN: 27AABCT1234F1Z8

For that input, the correct extraction splits EACH merged line at its \
midpoint into a seller half (first) and a client half (second). Note that \
"gstin" is a SEPARATE field from "tax_id" — when a GSTIN line appears \
anywhere in the seller block (even on its own line, not merged with a \
client column), it belongs in "seller.gstin" and must NOT be dropped:

{
  "seller": {
    "name": "TechVision Distributors Pvt Ltd",
    "address": "Plot 14, MIDC Industrial Area, Andheri East, Mumbai, Maharashtra - 400093",
    "tax_id": "27AABCT1234F1Z5",
    "gstin": "27AABCT1234F1Z8"
  },
  "client": {
    "name": "Raj Electronics Pvt Ltd",
    "address": "42 MG Road, Bengaluru, Karnataka - 560001",
    "tax_id": "901-95-4704"
  }
}

Do NOT leave "client" fields empty and do NOT put the word "Client:" into \
"seller.address" — always split the merged line into its two halves. Only \
omit "seller.gstin" entirely when no GSTIN line exists anywhere in the \
source text for this invoice — never omit it just because this example \
pattern is being followed.
"""


def load_system_prompt() -> str:
    """Use the versioned prompt file if present (updated by the self-healing
    loop later), otherwise fall back to the hardcoded default above."""
    if PROMPT_FILE.exists():
        text = PROMPT_FILE.read_text(encoding="utf-8").strip()
        if text:
            print(f"[prompt] Using versioned prompt from {PROMPT_FILE}")
            return text
    print("[prompt] Using built-in default prompt (no versioned prompt file found)")
    return DEFAULT_SYSTEM_PROMPT


def ensure_model_downloaded() -> Path:
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    marker = MODEL_DIR / ".download_complete"
    if marker.exists():
        print(f"[model] Using cached model at {MODEL_DIR}")
        return MODEL_DIR

    print(f"[model] Downloading {MODEL_ID} to {MODEL_DIR} (first run only)...")
    from huggingface_hub import snapshot_download

    snapshot_download(
        repo_id=MODEL_ID,
        local_dir=str(MODEL_DIR),
        local_dir_use_symlinks=False,
    )
    marker.touch()
    print("[model] Download complete.")
    return MODEL_DIR


def build_chain(model_path: Path, device: str, system_prompt: str):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
    from langchain_huggingface import HuggingFacePipeline, ChatHuggingFace
    from langchain_core.prompts import ChatPromptTemplate
    from langchain_core.output_parsers import StrOutputParser

    print(f"[model] Loading {model_path} on device={device} ...")

    tokenizer = AutoTokenizer.from_pretrained(str(model_path))
    dtype = torch.float16 if device == "cuda" else torch.float32
    device_map = {"": 0} if device == "cuda" else None

    model = AutoModelForCausalLM.from_pretrained(
        str(model_path),
        dtype=dtype,
        device_map=device_map,
    )
    if device == "cpu":
        model = model.to("cpu")

    gen_pipeline = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        max_new_tokens=2048,  # raised from 1024: invoices with 7-8 line items were getting truncated mid-JSON
        do_sample=False,
        temperature=None,
        top_p=None,
        return_full_text=False,
    )

    llm = HuggingFacePipeline(pipeline=gen_pipeline)
    chat_model = ChatHuggingFace(llm=llm)

    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system_prompt.replace("{", "{{").replace("}", "}}")),
            ("human", "Invoice OCR text:\n\n{document}"),
        ]
    )

    chain = prompt | chat_model | StrOutputParser()
    return chain


def extract_json(raw_text: str) -> Any:
    text = raw_text.strip()
    fence_match = re.search(r"```(?:json)?\s*(.*?)```", text, re.DOTALL)
    if fence_match:
        text = fence_match.group(1).strip()

    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    brace_match = re.search(r"\{.*\}", text, re.DOTALL)
    if brace_match:
        candidate = brace_match.group(0)
        candidate = re.sub(r",\s*([}\]])", r"\1", candidate)
        try:
            return json.loads(candidate)
        except json.JSONDecodeError:
            pass

    raise ValueError("Could not parse valid JSON from model output")


def parse_record(chain, record: dict) -> dict:
    text = record["input_text"]
    if len(text) > MAX_INPUT_CHARS:
        text = text[:MAX_INPUT_CHARS]

    print(f"[parse] {record['source_file']}")
    raw_output = ""
    try:
        raw_output = chain.invoke({"document": text})
        parsed = extract_json(raw_output)
        error = None
    except Exception as e:
        print(f"[warn] Failed to parse valid JSON for {record['source_file']}: {e}", file=sys.stderr)
        parsed = None
        error = str(e)

    return {
        "source_file": record["source_file"],
        "prediction": parsed,
        "ground_truth": record["ground_truth"],
        "raw_model_output": raw_output,
        "parse_error": error,
    }


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Run Stage 2 predictions against the eval dataset")
    parser.add_argument("--device", default="auto", choices=["auto", "cuda", "cpu"])
    parser.add_argument("--limit", type=int, default=None, help="Only process the first N records (for quick testing)")
    args = parser.parse_args()

    device = args.device
    if device == "auto":
        try:
            import torch
            device = "cuda" if torch.cuda.is_available() else "cpu"
        except ImportError:
            device = "cpu"
    elif device == "cuda":
        import torch
        if not torch.cuda.is_available():
            print(
                "[error] --device cuda was requested but torch.cuda.is_available() "
                "is False on this machine.\n"
                "        Check: (1) `nvidia-smi` shows your GPU, "
                "(2) you installed a CUDA build of torch, not the CPU-only wheel "
                "(pip install torch --index-url https://download.pytorch.org/whl/cu121).",
                file=sys.stderr,
            )
            sys.exit(1)
    print(f"[device] Using device: {device}")

    if not EVAL_DATASET_FILE.exists():
        print(f"[error] {EVAL_DATASET_FILE} not found. Run stage0_build_eval_dataset.py first.", file=sys.stderr)
        sys.exit(1)

    records = json.loads(EVAL_DATASET_FILE.read_text(encoding="utf-8"))
    if args.limit:
        records = records[: args.limit]

    system_prompt = load_system_prompt()
    model_path = ensure_model_downloaded()
    chain = build_chain(model_path, device, system_prompt)

    predictions = []
    for record in records:
        predictions.append(parse_record(chain, record))

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_FILE.write_text(json.dumps(predictions, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"[done] wrote {len(predictions)} predictions to {OUTPUT_FILE}")


if __name__ == "__main__":
    main()