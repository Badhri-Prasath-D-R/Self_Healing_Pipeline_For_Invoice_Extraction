import json
import re
import sys
from pathlib import Path
from typing import Any

MODEL_ID = "Qwen/Qwen2.5-1.5B-Instruct"

BASE_DIR = Path(__file__).resolve().parent
MD_DIR = BASE_DIR / "data" / "extracted_md"
OUTPUT_DIR = BASE_DIR / "data" / "structured_json"
OUTPUT_FILE = OUTPUT_DIR / "dataset.json"
MODEL_DIR = BASE_DIR / "models" / "qwen2.5-1.5b"

MAX_INPUT_CHARS = 6000

SYSTEM_PROMPT = """You are a strict information-extraction engine for invoices and \
similar business documents.

Read the document content provided by the user and extract every relevant \
field you can find into a single flat-ish JSON object. Do NOT force a fixed \
schema — invent key names that describe what you found (e.g. "vendor_name", \
"invoice_number", "invoice_date", "currency", "subtotal", "tax", "total", \
"line_items", "bill_to", "due_date", etc.). Common invoice fields:

- vendor / seller name and contact info
- invoice number / reference number
- invoice date and due date
- currency
- line items (as a JSON array of objects, each with whatever fields apply: \
description, quantity, unit_price, amount)
- subtotal, tax, discounts, total / amount due
- payment terms, PO number, if present

If the document is clearly NOT an invoice (e.g. a report, letter, contract), \
still extract whatever structured fields make sense for that document type \
(e.g. "document_type", "title", "date", "parties", "summary"), using your \
own judgement for key names.

Rules:
- Return ONLY a single valid JSON object. No prose, no markdown code fences, \
no explanations, no trailing commentary.
- Use null for a field you expect but cannot find. Omit fields that don't \
apply rather than inventing placeholders.
- Numbers (amounts, quantities) should be JSON numbers, not strings, when \
they are cleanly numeric.
"""

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

def build_chain(model_path: Path, device: str):
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
        max_new_tokens=1024,
        do_sample=False,
        temperature=None,
        top_p=None,
        return_full_text=False,
    )

    llm = HuggingFacePipeline(pipeline=gen_pipeline)
    chat_model = ChatHuggingFace(llm=llm)

    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", SYSTEM_PROMPT),
            ("human", "Document content:\n\n{document}"),
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

def parse_file(chain, md_path: Path) -> dict:
    text = md_path.read_text(encoding="utf-8")
    if len(text) > MAX_INPUT_CHARS:
        text = text[:MAX_INPUT_CHARS]

    print(f"[parse] {md_path.name}")
    raw_output = ""
    try:
        raw_output = chain.invoke({"document": text})
        parsed = extract_json(raw_output)
    except Exception as e:
        print(f"[warn] Failed to parse valid JSON for {md_path.name}: {e}", file=sys.stderr)
        parsed = {
            "_parse_error": str(e),
            "_raw_model_output": raw_output,
        }

    return {
        "source_file": md_path.name,
        "extraction": parsed,
    }

def main():
    import argparse

    parser = argparse.ArgumentParser(description="Parse extracted markdown into structured invoice JSON")
    parser.add_argument("--device", default="auto", choices=["auto", "cuda", "cpu"])
    args = parser.parse_args()

    device = args.device
    if device == "auto":
        try:
            import torch
            device = "cuda" if torch.cuda.is_available() else "cpu"
        except ImportError:
            device = "cpu"

    if not MD_DIR.exists():
        print(f"[error] Input directory not found: {MD_DIR}", file=sys.stderr)
        sys.exit(1)

    md_files = sorted(MD_DIR.glob("*.md"))
    if not md_files:
        print(f"[error] No markdown files found in {MD_DIR}", file=sys.stderr)
        sys.exit(1)

    model_path = ensure_model_downloaded()
    chain = build_chain(model_path, device)

    dataset = []
    for md_path in md_files:
        print(md_path,' ')
        record = parse_file(chain, md_path)
        dataset.append(record)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_FILE.write_text(json.dumps(dataset, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"[done] wrote {OUTPUT_FILE}")

if __name__ == "__main__":
    main()