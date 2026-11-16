# AGENTS.md — Drug Spoilage Detector

## Project Context

Hackathon entry for [Build Small Hackathon](https://huggingface.co/build-small-hackathon) — Track 1: **Backyard AI**.
Detects medicine spoilage from images of syrup bottles and drug packaging using a small VLM.
Submission deadline: **June 15, 2026**.

## Tech Stack

- **VLM Model:** `Qwen/Qwen2.5-VL-7B-Instruct` (7B params, runtime bitsandbytes INT4 quantized, served via HF Spaces ZeroGPU)
- **Orchestrator Model:** `deepseek-ai/DeepSeek-R1-Distill-Qwen-7B` (7B params, INT4, running on HF Spaces ZeroGPU)
- **Frontend:** Gradio 6.17.3 on HuggingFace Spaces
- **VLM Backend:** HuggingFace Spaces ZeroGPU (T4) with vLLM inference
- **Orchestrator Backend:** HuggingFace Spaces ZeroGPU (T4) with transformers inference
- **Python:** 3.13
- **Charts:** Plotly (interactive, renders in `gr.Plot`)
- **Entry point:** `app.py` (declared in README.md frontmatter)

## Architecture

```
HF Spaces (Gradio)                        HF Spaces ZeroGPU (T4)
┌──────────────────────┐         HTTPS POST ┌──────────────────────┐  HTTPS POST ┌──────────────────────┐
│  app.py              │         JSON resp  │  vlm_client.py       │  JSON resp  │  DeepSeek-R1-        │
│  orchestrator.py  ───┼──────────┬────────►│  vLLM + Qwen2.5-     │────────────►│  Distill-Qwen-7B     │
│  (calls engine.py)   │          │        │  VL-7B INT4         │             │  INT4                │
│  llm_client.py ──────┼──────────┘        │  @spaces/ZeroGPU    │◄────────────│  @spaces/ZeroGPU     │
│  prompts.py          │◄─────────┴─────────│  scales to 0       │             │  synthesis only       │
│  utils.py            │                     └──────────────────────┘             └──────────────────────┘
│  visualization.py    │
└──────────────────────┘
```

- **Frontend** (HF Spaces): Gradio UI, chart generation, spoilage scoring — no GPU needed
- **Backend** (HF Spaces ZeroGPU): VLM inference only — spins up on request, scales to 0 when idle
- **Communication:** Base64-encoded JPEG images sent via HTTPS OpenAI-compatible API to the VLM ZeroGPU endpoint (single image, cropped to label via ImageEditor)

## Hackathon Constraints

| Rule | Value |
|------|-------|
| Max model params | ≤ 32B (we use 8B, INT4 quantized) |
| App type | Gradio on HuggingFace Spaces |
| Submission | Space link + demo video + social post |
| No cloud APIs | Inference runs on HF Spaces ZeroGPU (not a cloud API service like OpenAI) |

### Merit Badges Targeted

- **Off the Grid** — inference runs on HF Spaces ZeroGPU (no third-party API service)
- **Sharing is Caring** — publish agent trace to HF Hub
- **Off-Brand** — custom Gradio theme via `gr.themes`

## File Architecture

```
app.py              # Gradio Blocks UI — main entry point (HF Spaces)
backend.py          # SageMaker deployment: vLLM + Qwen2.5-VL-7B INT4 on SageMaker GPU
src/
├── __init__.py
├── orchestrator.py # Orchestrator class + analyze_image() drop-in replacement
├── llm_client.py   # Deepseek-R1 client with reasoning strip
                     (remote via ZeroGPU endpoint or local transformers)
├── vlm_client.py   # HF ZeroGPU VLM client (OpenAI-compatible /v1/chat/completions)
├── engine.py       # HTTPS client that calls VLM endpoint (two-pass pipeline)
                     (low-level VLM client, now called by the orchestrator)
├── prompts.py      # VLM prompt templates with OCR context injection
├── utils.py        # Date parsing, chemical reference DB, spoilage scoring
└── visualization.py # Plotly charts: chemical bar, timeline, gauge, radar
+── deploy_sagemaker.py  # SageMaker deployment script (deprecated — replaced by vlm_client.py)
tests/
├── __init__.py
├── test_endpoint.py    # HF ZeroGPU VLM endpoint connectivity test
├── test_improvements.py # OpenBMB optimization tests
└── test_pipeline.py    # Full pipeline integration test
requirements.txt    # Lightweight: requests, gradio, plotly, pandas, etc.
README.md           # HF Spaces frontmatter (already configured)
AGENTS.md           # This file
agent_trace/        # Published trace for Sharing is Caring badge
```

## VLM Analysis Pipeline

Two-pass architecture for better text extraction from medicine packaging:

**Pass 1 — OCR Extraction:**
1. **PROMPT_OCR** — Read ALL visible text from every surface (front label, side panel, back, stickers, cap, foil). Returns structured JSON with text blocks, packaging type, and visible surfaces. Focuses on dot-matrix/stamped text for batch numbers and dates.

**Pass 2 — Structured Analysis** (uses OCR text as context):
1. **PROMPT_INFO** — Extract medicine info (name, manufacturer, dates, ingredients, storage). Includes date-guessing warning to set null when dates aren't visible.
2. **PROMPT_SPOILAGE** — Assess visual spoilage with pharmacological context rule to avoid false positives on suspensions.
3. **PROMPT_BACTERIA** — Bacteria growth estimate with preservative analysis from text. Ignores natural suspension opacity.
4. **PROMPT_CHEMICALS** — List chemicals with quantity, category, and risk level

All prompts instruct the model to return **valid JSON only**. Regex fallback parser handles non-JSON outputs. The OCR text is injected into Pass 2 prompts via `{OCR_TEXT}` placeholder.

**Optimization:** Pass 2 runs all 4 prompts in parallel via `ThreadPoolExecutor` (~4x speedup).

**Prompt format:** Prompts use `{OCR_TEXT}` placeholder which is filled via Python's `.format()`. JSON examples in prompts must use escaped braces `{{ }}` to avoid `KeyError`.

**Stage 3 — Deepseek R1 Synthesis:** After Pass 2 completes, the VLM JSON outputs are handed to the orchestrator stage (`deepseek-ai/DeepSeek-R1-Distill-Qwen-7B` on HF Spaces ZeroGPU) for a **single synthesis call**. It reviews VLM output, corrects errors, and emits a final verdict + reasoning + confidence. Design principle: **no multi-step tool-calling loops** — VLM JSON is precomputed first, then one LLM call synthesizes it. This keeps latency acceptable on a 7B-class model. The Deepseek-R1 `...` chain-of-thought blocks are stripped by `strip_reasoning()` in `src/llm_client.py` before JSON parsing.

## Two-Mode Date System

The app operates in two modes based on whether dates are visible on the medicine label:

| | Mode 1 (dates found) | Mode 2 (no dates) |
|---|---|---|
| **Primary verdict** | Static expiry date | Visual spoilage score |
| **Timeline shows** | MFG → Today → Expiry | Today → Visual estimate |
| **Predicted spoilage** | Secondary note only | Primary estimate |
| **Verdict wording** | "X days until expiry" | "Visually OK / Spoiled" |
| **Use case** | Normal purchase | Shopkeeper cut strip |

**Mode detection:** If `exp_date` is found → Mode 1. If `exp_date` is null → Mode 2.

## Spoilage Scoring

```
score = (visual_indicators * 0.35) + (bacteria_level * 0.25) + (date_proximity * 0.20) + (color_deviation * 0.10) + (dynamic_expiry * 0.10)
```

Thresholds: >60 = SPOILED, 30-60 = WARNING, <30 = SAFE

## Date Parsing

`parse_date()` in `utils.py` handles multiple formats:
- DD/MM/YYYY, DD-MM-YYYY, YYYY-MM-DD, DD.MM.YYYY
- MM/YYYY, MM-YYYY, YYYY only
- Month YYYY (e.g., "SEP 2025"), DD Month YYYY
- **Dot-normalized:** "SEP.2025" → "SEP 2025" (dots replaced with spaces before parsing)
- **Pattern priority:** Month YYYY comes before YYYY-only to prevent incorrect matching

## Python Fallback Calculations

When VLM can't extract certain data, Python calculates:
1. **Bacteria growth curve** — Logistic model based on ingredients, preservatives, spoilage level
2. **Color degradation timeline** — Exponential decay based on visual indicators
3. **Dynamic expiry** — Adjusted shelf life based on visual damage, preservatives, color deviation

## Key Gotchas

- Model repo is `Qwen/Qwen2.5-VL-7B-Instruct`.
- vLLM requires `--quantization bitsandbytes --load-format bitsandbytes` for INT4 models (configured in the HF Spaces ZeroGPU container).
- VLM endpoint URL is set via `VLM_ENDPOINT_URL` environment variable (HF Spaces ZeroGPU account required).
- HF credentials required (via `TOKEN` env var).
- Deepseek R1 emits chain-of-thought in `...` blocks; `strip_reasoning()` removes them before JSON parsing.
- Deepseek R1 synthesis is a single call, not a tool-calling loop (latency budget).
- ZeroGPU runs on HF Spaces; `ZEROGPU_ENDPOINT_URL` env var required.
- Qwen2.5-VL-7B INT4 (~5GB VRAM) + Deepseek-R1-Distill-Qwen-7B INT4 (~5GB) fit on an L4 (24GB) / ZeroGPU T4 (16GB) with quantization.
- Base64 encoding increases image payload ~33% — JPEG quality 95 preserves fine print.
- HF Spaces ZeroGPU cold start variability: first request may be slow after idle periods.
- vLLM serves an OpenAI-compatible API — no custom client code needed.
- Plotly charts render in `gr.Plot()` components — not `gr.Image()`.
- The VLM backend (`vlm_client.py`) is deployed as a separate HF Spaces ZeroGPU Space from the frontend.
- Image preprocessing enhances contrast + sharpens for medicine labels.
- Using `uv` instead of `pip` for faster Space builds.
- OCR text truncated to 800 chars in Pass 2 prompts (up from 300) to capture more date/ingredient info.
- OCR exceptions now propagate to error handler instead of failing silently.
- Dynamic expiry defaults to 1-year shelf life when dates are missing.
- `format_ocr_for_prompt` max_chars=800 for better context.
- Pass 2 prompts use `ThreadPoolExecutor` for parallel execution.
- User text (from "Additional Info" box) takes priority over OCR text in prompts.
- Chemical chart shows Name → Quantity with bar width proportional to parsed values.
- `.env` file contains `TOKEN` (HF token) and `VLM_ENDPOINT_URL` (VLM endpoint).
- HF Spaces ZeroGPU deployment requires HF credentials and `huggingface_hub` configured.

## Running Locally

```bash
# Frontend only (needs VLM_ENDPOINT_URL set)
pip install -r requirements.txt
VLM_ENDPOINT_URL=https://your-vlm-space.hf.space python app.py
# Opens at http://localhost:7860
```

## Deploying

### 1. Deploy VLM backend Space (ZeroGPU)

Create a separate HuggingFace Space with ZeroGPU hardware that serves `Qwen/Qwen2.5-VL-7B-Instruct` (INT4) behind an OpenAI-compatible endpoint. Save the resulting URL.

```bash
# In the VLM Space, expose an OpenAI-compatible chat endpoint
# Requires HF Spaces ZeroGPU (T4) hardware
```

The VLM service exposes an OpenAI-compatible endpoint; `src/vlm_client.py` calls it via `VLM_ENDPOINT_URL`.

### 2. Deploy Deepseek R1 orchestrator Space (ZeroGPU)

Create a separate HuggingFace Space with ZeroGPU hardware that serves `deepseek-ai/DeepSeek-R1-Distill-Qwen-7B` (INT4) behind an OpenAI-compatible endpoint. Save the resulting URL.

```bash
# In the orchestrator Space, expose an OpenAI-compatible chat endpoint
# Requires HF Spaces ZeroGPU (T4) hardware
```

The orchestration service exposes an OpenAI-compatible endpoint; `src/llm_client.py` calls it via `ZEROGPU_ENDPOINT_URL`.

### 3. Deploy HF Space frontend

1. Set `VLM_ENDPOINT_URL` as a Space secret (Settings → Secrets)
2. Set `ZEROGPU_ENDPOINT_URL` as a Space secret (Settings → Secrets)
3. Push code to the HF Space
4. README.md frontmatter already declares `sdk: gradio`, `app_file: app.py`

## Submission Checklist

- [x] VLM backend deployed and endpoint URL saved
- [x] HF Space deployed with VLM_ENDPOINT_URL secret configured
- [ ] Demo video recorded (show: upload image → crop label → analyze → charts + verdict)
- [ ] Social media post made
- [ ] Agent trace published to HF Hub (Sharing is Caring badge)
