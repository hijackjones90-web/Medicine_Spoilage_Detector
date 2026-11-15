# AGENTS.md — Drug Spoilage Detector

## Tech Stack

- **Model:** Qwen2.5-VL 7B (7B params, 4-bit quantized) — free cloud GPU via Kaggle/Colab, or local Ollama
- **Frontend:** Gradio 4.+ on HuggingFace Spaces
- **Backend:** Kaggle GPU notebook (30 GPU-hrs/week free) OR local Ollama CPU
- **Python:** 3.11/3.13
- **Charts:** Plotly interactive charts in `gr.Plot()` components

## File Architecture

```
app.py              # Gradio Blocks UI — main entry point
src/
├── engine.py       # Two-pass VLM pipeline (OCR → structured analysis with retries)
├── prompts.py      # Prompt templates with {OCR_TEXT} placeholder; escaped braces `{{ }}`
├── utils.py        # Date parsing, chemical reference DB, spoilage scoring
└── visualization.py # Plotly charts: chemical bar, timeline, gauge, radar, growth curve
tests/
├── test_endpoint.py    # VLM endpoint connectivity test
├── test_improvements.py # VLM improvement tests
└── test_pipeline.py   # Full pipeline integration test
requirements.txt    # requests, gradio, plotly, pandas, numpy, python-dateutil, python-dotenv, ollama
README.md           # HF Spaces frontmatter (already configured)
AGENTS.md           # This file
.env.examples       # Example env file — copy to .env and fill in secrets
```

## VLM Backend Options (pick one)

### Option 1 — Kaggle GPU (recommended)

Kaggle provides free GPU (P100 or 2xT4) with a predictable ~30 GPU-hours/week quota. No credit card required — just phone verification.

Setup:
1. Create free Kaggle account and verify phone number
2. Create new Kaggle notebook with GPU accelerator (P100 or T4)
3. Install vLLM: `!pip install vllm==0.6.6 transformers==4.45.2 bitsandbytes`
4. Run vLLM server on Kaggle notebook:
   ```bash
   vllm serve Qwen/Qwen2.5-VL-7B-Instruct \
     --port 8000 --dtype auto --quantization bitsandbytes \
     --load-format bitsandbytes --host 0.0.0.0 \
     --limit-mm-per-prompt image=8
   ```
5. Copy the notebook's public URL as `KAGGLE_VLM_URL` in `.env`
6. Alternatively use `Qwen/Qwen2-VL-7B` or `openbmb/MiniCPM-V-2_6` as the VLM model

**Caveat:** Sessions max out at ~12 hours. Save checkpoints; restart when quota resets weekly.

### Option 2 — Google Colab Free Tier

Free T4 GPU with dynamic usage limits. Less predictable than Kaggle but no phone verification required.

1. Open `https://colab.research.google.com/` — new notebook
2. Runtime → Change runtime type → T4 GPU
3. Install vLLM and run server
4. Use Colab's public URL as `KAGGLE_VLM_URL`

### Option 3 — Local Ollama (CPU-only laptop)

For slow but free local inference (no cloud dependency):

1. Install Ollama: `curl -fsSL https://ollama.com/install.sh | sh`
2. Pull a VLM: `ollama pull qwen2.5vl:7b` (6 GB download)
3. Run: `ollama serve` in background
4. Set `OLLAMA_HOST=http://localhost:11434` in `.env`

**Caveat:** CPU-only VLMs are slow (~1–5 tok/s). Expect 30–60+ seconds per image analysis.

### Option 4 — Hugging Face ZeroGPU Spaces (frontend only)

Free CPU Spaces for hosting the Gradio demo. ZeroGPU (shared GPU minutes) available for inference but shared/queued. Best for showing the demo, not production inference.

## Two-Pass VLM Pipeline

**Pass 1 — OCR:** `PROMPT_OCR` reads ALL visible text from every surface. Returns JSON with text blocks, packaging type, surfaces visible.

**Pass 2 — Structured Analysis (runs 4 prompts in parallel via ThreadPoolExecutor):**
- `PROMPT_INFO` — Extract medicine info (name, dates, ingredients, storage). Sets fields to `null` if not found — **do not guess**.
- `PROMPT_SPOILAGE` — Assess visual spoilage; includes suspension context rule (naturally opaque ≠ cloudiness).
- `PROMPT_BACTERIA` — Bacteria growth estimate with preservative analysis.
- `PROMPT_CHEMICALS` — List chemicals with quantity, category, risk level.

All prompts instruct model to return **valid JSON only**. Regex fallback parser in `engine.py:_extract_json()` handles non-JSON outputs.

**Key:** User text from "Additional Info" box takes priority over OCR text in prompts (engine.py:245-247). OCR text truncated to 800 chars via `format_ocr_for_prompt()` (prompts.py:12).

## Prompt Format

Prompts use `{OCR_TEXT}` placeholder filled via Python `.format()`. JSON examples in prompts must use **escaped braces `{{ }}`** to avoid `KeyError`. Example:

```python
prompt_template.format(OCR_TEXT=combined)  # NOT .format(OCR_TEXT=combined) without escaping
```

## Date Parsing (`utils.py:parse_date()`)

Handles: DD/MM/YYYY, DD-MM-YYYY, YYYY-MM-DD, DD.MM.YYYY, MM/YYYY, MM-YYYY, YYYY, Month YYYY, DD Month YYYY.

**Gotchas:**
- Dot-normalized: `"SEP.2025"` → `"SEP 2025"` (dots → spaces before parsing)
- Pattern priority: `Month YYYY` comes before `YYYY-only` to prevent incorrect matching

## Spoilage Scoring (`utils.py:calculate_spoilage_score()`)

```
score = visual(35%) + bacteria(25%) + date(20%) + color(10%) + dynamic_expiry(10%)
Thresholds: >60 = SPOILED, 30-60 = WARNING, <30 = SAFE
```

When VLM can't extract data, Python calculates fallback: bacteria growth curve (logistic model), color degradation timeline (exponential decay), dynamic expiry (adjusted shelf life).

## Key Gotchas (would miss without help)

- vLLM requires `--quantization bitsandbytes --load-format bitsandbytes` for INT4 models on cloud GPU
- Kaggle GPU sessions max out ~12 hours; quota resets weekly (~30 GPU-hrs)
- Base64 encoding increases image payload ~33% — JPEG quality 95 preserves fine print
- Local Ollama CPU inference is slow (~1–5 tok/s)
- Plotly charts render in `gr.Plot()` — **not** `gr.Image()`
- Cloud GPU backend (Kaggle/Colab) is separate from HF Space frontend
- Image preprocessing enhances contrast + sharpens for medicine labels
- OCR exceptions propagate to error handler instead of failing silently
- Dynamic expiry defaults to 1-year shelf life when dates are missing
- Pass 2 prompts use `ThreadPoolExecutor` for parallel execution (~4x speedup)
- User text priority over OCR text in prompts
- Chemical chart: Name → Quantity with bar width proportional to parsed values
- JSON examples in prompts use escaped braces `{{ }}` to avoid `KeyError`
- Set `KAGGLE_VLM_URL` env var for cloud GPU, or `OLLAMA_HOST` for local
- If neither is set, app returns error: "No VLM endpoint configured"

## Running Locally

### Kaggle cloud GPU (recommended)

```bash
pip install -r requirements.txt
KAGGLE_VLM_URL=https://your-kaggle-username.notebooks.kaggle.io/api/v1 python app.py
# Opens at http://localhost:7860
```

### Local Ollama (CPU-only)

```bash
pip install -r requirements.txt
OLLAMA_HOST=http://localhost:11434 OLLAMA_VLM_MODEL=qwen2.5vl:7b python app.py
# Opens at http://localhost:7860
```

## Deploying

### 1. Deploy VLM backend on Kaggle GPU

1. Create Kaggle account, verify phone number
2. Create notebook with GPU (P100 or T4), run vLLM server
3. Copy notebook public URL as `KAGGLE_VLM_URL` in `.env`
4. Deploy HG Space and set `KAGGLE_VLM_URL` as a Space secret

### 2. Deploy HF Space frontend

1. Set `KAGGLE_VLM_URL` as a Space secret (Settings → Secrets)
2. Push code to the HF Space
3. README.md frontmatter already declares `sdk: gradio`, `app_file: app.py`

## Testing

```bash
# Test cloud GPU endpoint
python tests/test_endpoint.py

# Test full pipeline
python tests/test_pipeline.py
```

## Submission Checklist

- [ ] Kaggle GPU deployed with vLLM serving VLM
- [ ] HF Space deployed with KAGGLE_VLM_URL secret configured
- [ ] Demo video recorded (show: upload image → analyze → charts + verdict)
- [ ] Social media post made
- [ ] Agent trace published to HF Hub (Sharing is Caring badge)