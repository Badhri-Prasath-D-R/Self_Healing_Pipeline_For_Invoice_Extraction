import argparse
import sys
from pathlib import Path

MODEL_ID = "opendatalab/MinerU2.5-Pro-2604-1.2B"

BASE_DIR = Path(__file__).resolve().parent
RAW_PDF_DIR = BASE_DIR / "data" / "raw_pdfs"
OUTPUT_DIR = BASE_DIR / "data" / "extracted_md"
MODEL_DIR = BASE_DIR / "models" / "mineru2.5-pro"



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


def load_client(model_path: Path, device: str):
    import torch
    from transformers import AutoProcessor, Qwen2VLForConditionalGeneration
    from mineru_vl_utils import MinerUClient

    print(f"[model] Loading model from {model_path} on device={device} ...")

    if device == "cpu":
        dtype = "float32"
        device_map = None
    else:
        dtype = torch.float16
        device_map = {"": 0}  # pin everything to cuda:0, avoid "auto" sharding overhead

    model = Qwen2VLForConditionalGeneration.from_pretrained(
        str(model_path),
        dtype=dtype,
        device_map=device_map,
    )
    if device == "cpu":
        model = model.to("cpu")

    processor = AutoProcessor.from_pretrained(str(model_path), use_fast=True)

    client = MinerUClient(
        backend="transformers",
        model=model,
        processor=processor,
        image_analysis=False,  # set True if you also want chart/image descriptions
    )
    return client

def pdf_to_images(pdf_path: Path):
    import fitz
    from PIL import Image
    import io

    doc = fitz.open(str(pdf_path))
    images = []
    zoom = 2.0
    mat = fitz.Matrix(zoom, zoom)
    for page in doc:
        pix = page.get_pixmap(matrix=mat)
        img = Image.open(io.BytesIO(pix.tobytes("png")))
        images.append(img)
    doc.close()
    return images


def extract_pdf(client, pdf_path: Path, output_dir: Path):
    print(f"[extract] {pdf_path.name}")
    images = pdf_to_images(pdf_path)

    page_markdowns = []
    for i, img in enumerate(images, start=1):
        print(f"  - page {i}/{len(images)}")
        result = client.two_step_extract(img)
        md = getattr(result, "to_markdown", None)
        page_md = md() if callable(md) else str(result)
        page_markdowns.append(f"<!-- page {i} -->\n\n{page_md}")

    full_md = "\n\n---\n\n".join(page_markdowns)

    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / f"{pdf_path.stem}.md"
    out_path.write_text(full_md, encoding="utf-8")
    print(f"[extract] wrote {out_path}")
    return out_path


def main():
    parser = argparse.ArgumentParser(description="Extract markdown from PDFs via MinerU2.5-Pro")
    parser.add_argument("pdf", nargs="?", help="Single PDF filename (in data/raw_pdfs/) to process. If omitted, processes all PDFs.")
    parser.add_argument("--device", default="auto", choices=["auto", "cuda", "cpu"], help="Device to run the model on")
    args = parser.parse_args()
    device = args.device
    if device == "auto":
        try:
            import torch
            device = "cuda" if torch.cuda.is_available() else "cpu"
        except ImportError:
            device = "cpu"

    if not RAW_PDF_DIR.exists():
        print(f"[error] Input directory not found: {RAW_PDF_DIR}", file=sys.stderr)
        sys.exit(1)

    if args.pdf:
        pdf_files = [RAW_PDF_DIR / args.pdf]
        if not pdf_files[0].exists():
            print(f"[error] File not found: {pdf_files[0]}", file=sys.stderr)
            sys.exit(1)
    else:
        pdf_files = sorted(RAW_PDF_DIR.glob("*.pdf"))
        if not pdf_files:
            print(f"[error] No PDFs found in {RAW_PDF_DIR}", file=sys.stderr)
            sys.exit(1)

    model_path = ensure_model_downloaded()
    client = load_client(model_path, device)


    for pdf_path in pdf_files:
        try:
            print(pdf_path,' ')
            extract_pdf(client, pdf_path, OUTPUT_DIR)
        except Exception as e:
            print(f"[error] Failed on {pdf_path.name}: {e}", file=sys.stderr)

    print("[done]")

if __name__ == "__main__":
    main()