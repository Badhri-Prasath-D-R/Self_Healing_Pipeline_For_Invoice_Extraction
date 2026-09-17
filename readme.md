# Self-Healing Invoice Extraction Pipeline

A local, GPU-accelerated pipeline that extracts structured JSON from invoice
PDFs using a small local LLM (Qwen2.5-1.5B-Instruct), and automatically
improves its own extraction prompt over time by learning from its real
mistakes — without any model fine-tuning or retraining.

Built as a college MLOps project (Semester 5).

## What this actually does

1. Takes invoice OCR text and extracts it into a fixed structured JSON
   schema (invoice number, seller, client, line items, totals) using a
   local 1.5B-parameter LLM.
2. Compares its own output against ground truth to find real, concrete
   mistakes (missing fields, wrong values, JSON parse failures).
3. Automatically builds corrected few-shot examples from those mistakes
   and tests whether adding them to the prompt actually helps — on
   documents it has never seen, not just the ones the mistakes came from.
4. If the improved prompt is genuinely better, it's promoted to production
   automatically. If not, it's discarded. Either way, the full experiment
   (old prompt, new prompt, exemplars used, before/after accuracy) is
   logged to MLflow.
5. The invoice dataset, model weights, and every pipeline output are
   version-controlled with DVC alongside the code in git, so any run can
   be traced back to exactly which data and prompt version produced it.

**No model weights are fine-tuned anywhere in this pipeline.** Only the
system prompt (plain text) is versioned and improved. A future extension
(see "Planned / Future Work" below) would add LoRA/QLoRA adapter training
on top of this, but that has not been built yet.

## Results

On a 100-invoice dataset (Kaggle "High-Quality OCR-Ready Invoice PDFs"),
split 80 train / 20 holdout:

| Metric (holdout set) | Before self-healing | After 1 round |
|---|---|---|
| Field-level accuracy | 72.3% | **87.5%** |
| Exact document match | 0.0% | **65.0%** |

The improvement was found and validated entirely automatically: the loop
identified that seller/client address-column splitting was the dominant
error, built 3 few-shot correction examples from real training-set
mistakes, and confirmed the fix generalized to unseen holdout documents
before promoting it.

## Architecture

```
Kaggle CSV (ground truth + OCR text)
        |
        v
 stage0_build_eval_dataset.py   ->  data/eval_dataset/dataset.json
        |                          (80/20 train/holdout split, seeded)
        v
 stage2_parse.py                ->  data/structured_json/predictions.json
        |   (local Qwen2.5-1.5B,     (model's extraction attempt per invoice)
        |    prompt from
        |    prompts/current_system_prompt.txt,
        |    falls back to a built-in default)
        v
 stage3_find_mistakes.py        ->  data/mistakes/mistakes.json
        |                          data/mistakes/summary.json
        |                          (field-by-field diff vs ground truth,
        |                           per-split accuracy breakdown)
        v
 stage4_prompt_tuner.py         ->  prompts/current_system_prompt.txt
        |                          (updated, only if holdout improved)
        |                          + full run logged to MLflow
        +--------------+
                        v
              MLflow tracking server
        (old/new prompt text, exemplars used,
         holdout accuracy before/after, promotion decision)
```

`stage1_ingest.py` (MinerU2.5-Pro OCR) exists and works standalone, but the
pipeline above currently uses the dataset's own pre-supplied OCR text
(`ocred_text` column) as a shortcut, documented in `stage0`'s docstring.
Swapping in Stage 1's real output is a drop-in change (see "Using your own
PDFs" below).

## Prerequisites

- Windows with an NVIDIA GPU (developed on a 6GB laptop GPU; also runs on
  CPU, much slower)
- Python 3.11, a virtual environment
- NVIDIA drivers + a CUDA-enabled PyTorch build (see Setup step 2)
- ~10 GB free disk space (model weights + dataset + DVC cache)

## Setup (one-time)

### 1. Python environment

```
python -m venv venv
venv\Scripts\activate
pip install langchain langchain-huggingface transformers huggingface_hub
pip install mlflow dvc
pip install pymupdf pillow
```
(pymupdf/pillow only needed if you use stage1_ingest.py)

### 2. PyTorch with CUDA (skip if using CPU only)

Check your driver supports CUDA:
```
nvidia-smi
```

Install the CUDA build (cu121 works with most recent drivers):
```
pip uninstall torch torchvision torchaudio
pip install torch --index-url https://download.pytorch.org/whl/cu121
```

Verify:
```
python -c "import torch; print(torch.__version__); print(torch.cuda.is_available())"
```
Should print `True`.

### 3. Dataset

Download the Kaggle dataset "High-Quality OCR-Ready Invoice PDFs" and
place it at:
```
data/kaggle_invoices/Batch 1/batch_1.csv
data/kaggle_invoices/Batch 1/invoices/*.pdf
```

### 4. MLflow tracking server

In its own terminal (leave running):
```
mlflow server --backend-store-uri sqlite:///mlflow.db --default-artifact-root ./mlruns --host 127.0.0.1 --port 5000 --workers 1
```
Confirm it's reachable at http://127.0.0.1:5000 in a browser.

(`--workers 1` avoids a Windows-specific uvicorn multi-worker socket issue.
If you don't pass it and see startup errors mentioning `WinError 10022`,
check the UI anyway; one worker usually still starts successfully.)

### 5. DVC (data + model versioning)

```
dvc init
git add .dvc .dvcignore
git commit -m "Initialize DVC"

dvc remote add -d localstorage "<path outside the project, e.g. C:\Users\you\Desktop\dvc_storage>"
git add .dvc/config
git commit -m "Configure DVC local remote"

dvc add "data\kaggle_invoices"
dvc add "models"
git add "data\kaggle_invoices.dvc" "models.dvc" "data\.gitignore" ".gitignore"
git commit -m "Track dataset and models with DVC"
dvc push
```

## Running the pipeline

Make sure the MLflow server (step 4 above) is running in its own terminal
throughout. In a second terminal, activate the venv and run each stage in
order, one at a time:

```
python stage0_build_eval_dataset.py
```
Builds `data/eval_dataset/dataset.json` from the Kaggle CSV, with a fixed
80/20 train/holdout split.

```
python stage2_parse.py --device cuda
```
Runs all 100 documents through the LLM using whatever prompt is currently
active (`prompts/current_system_prompt.txt` if it exists, otherwise a
built-in default). Downloads Qwen2.5-1.5B-Instruct on first run (cached
after that). Use `--device cpu` if you don't have a CUDA GPU, or
`--limit 5` to test on a handful of documents first.

```
python stage3_find_mistakes.py
```
Compares predictions against ground truth, field by field. Writes
`data/mistakes/mistakes.json` (full diffs) and `data/mistakes/summary.json`
(aggregate stats, most common error fields, train/holdout accuracy).

```
python stage4_prompt_tuner.py --device cuda
```
The self-healing step. Reads the mistakes found in step 3, builds few-shot
correction examples targeting the current most common error pattern,
evaluates the resulting prompt against the holdout set (which the
corrections were never built from), and promotes it to
`prompts/current_system_prompt.txt` only if holdout field accuracy
improves by at least 1 percentage point. Logs everything to MLflow.

Check the result in the MLflow UI (http://127.0.0.1:5000) under the
`invoice_prompt_self_healing` experiment: old vs. new prompt text, which
documents were used as exemplars, and both accuracy scores.

### Repeating the loop

To find the next round of mistakes under the (possibly now-updated)
prompt and try to improve further:
```
python stage2_parse.py --device cuda
python stage3_find_mistakes.py
python stage4_prompt_tuner.py --device cuda
```

### After a run: commit the new state

```
dvc add "data\eval_dataset" "data\structured_json" "data\mistakes" "prompts"
git add "data\eval_dataset.dvc" "data\structured_json.dvc" "data\mistakes.dvc" "prompts.dvc" "data\.gitignore"
git commit -m "Describe what changed, e.g. round 2 of self-healing loop"
dvc push
```

## Using your own PDFs instead of the Kaggle dataset

1. Place PDFs in `data/raw_pdfs/`.
2. Run Stage 1 to OCR them:
   ```
   python stage1_ingest.py --device cuda
   ```
   Downloads MinerU2.5-Pro-2604-1.2B on first run. Output goes to
   `data/extracted_md/*.md`.
3. You'll need ground-truth JSON for each invoice to evaluate against,
   either label them by hand, or adapt `stage0_build_eval_dataset.py` to
   read from `data/extracted_md/` instead of the Kaggle CSV once you have
   labels.

## Project structure

```
data/
  kaggle_invoices/        DVC-tracked: raw dataset (PDFs + CSV)
  raw_pdfs/                (only used if running Stage 1 on your own PDFs)
  extracted_md/             Stage 1 output (OCR markdown)
  eval_dataset/            DVC-tracked: Stage 0 output (input + ground truth + split)
  structured_json/         DVC-tracked: Stage 2 output (model predictions)
  mistakes/                DVC-tracked: Stage 3 output (diffs + summary)
models/
  mineru2.5-pro/           DVC-tracked: OCR model weights (cached after first download)
  qwen2.5-1.5b/            DVC-tracked: extraction model weights (cached after first download)
prompts/
  current_system_prompt.txt  DVC-tracked: the currently-promoted prompt (written by Stage 4)
mlflow.db, mlruns/          MLflow's local tracking store
stage0_build_eval_dataset.py
stage1_ingest.py
stage2_parse.py
stage3_find_mistakes.py
stage4_prompt_tuner.py
```

## Known limitations

- **6GB VRAM is tight** for running two model instances back-to-back
  (Stage 4 evaluates an old and new prompt in sequence); the code
  explicitly frees GPU memory between the two passes and retries once on
  a transient out-of-memory error, but very large batches may still need
  `--device cpu` on smaller GPUs.
- **The exact-match metric is strict**: every one of 20+ fields per
  invoice must match exactly. Field-level accuracy is the primary metric
  used for promotion decisions for this reason.
- **The eval dataset is synthetic** (Kaggle-generated invoices). Prompt
  improvements found here are a proof of the mechanism, not a guarantee
  they'll transfer as-is to real-world scanned invoices with messier OCR.
- **Greedy decoding** (`do_sample=False`) is used throughout for
  reproducibility, so results are deterministic given the same prompt and
  input; useful for honest before/after comparisons, but means the model
  never explores alternative phrasings on its own.

## Planned / Future Work

- **QLoRA adapter training**: once the prompt-healing loop has accumulated
  enough real drift examples, train a small LoRA/QLoRA adapter on top of
  Qwen2.5-1.5B for cases a prompt alone can't fix. Deliberately built
  *after* the prompt loop, since it's higher-cost and the prompt-only
  approach already recovers most of the accuracy gap.
- **Frontend**: a simple UI to trigger pipeline runs and browse
  mistakes/prompt history without the command line.
- **Real PDFs via Stage 1**: replace the Kaggle dataset's pre-supplied OCR
  text with MinerU's actual output, to test whether the self-healing loop
  handles real OCR noise (not just the clean synthetic text used so far).