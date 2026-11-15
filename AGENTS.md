# AGENTS.md — Drug Spoilage Detector

## Tech Stack

- **Model:** `openbmb/MiniCPM-V-2_6-int4` (8B params, INT4 quantized via bitsandbytes, served via vLLM on Modal GPU L4)
- **Frontend:** Gradio 4.+ on HuggingFace Spaces
- **Backend:** Modal serverless GPU — spins up on demand, scales to 0 when idle
- **Python:** 3.11/3.13
- **Charts:** Plotly interactive charts in `gr.Plot()` components

## File Architecture

```
app.py              # Gradio Blocks UI — main entry point
backend.py          # Modal deployment: vLLM + MiniCPM-V 2.6 INT4 on L4 GPU
src/
├── engine.py       # Two-pass VLM pipeline (OCR → structured analysis with retries)
├── prompts.py      # Prompt templates with {OCR_TEXT} placeholder; escaped braces `{{ }}`
├── utils.py        # Date parsing, chemical reference DB, spoilage scoring
└── visualization.py # Plotly charts: chemical bar, timeline, gauge, radar, growth curve
tests/              # pytest unit/integration tests
requirements.txt    # requests, gradio, plotly, pandas, numpy, python-dateutil, python-dotenv
```

## Two-Pass VLM Pipeline

**Pass 1 — OCR:** `PROMPT_OCR` reads ALL visible text from every surface. Returns JSON with text blocks, packaging type, surfaces visible.

**Pass 2 — Structured Analysis (runs 4 prompts in parallel via ThreadPoolExecutor):**
- `PROMPT_INFO` — Extract medicine info (name, dates, ingredients, storage). Sets fields to `null` if not found — **do not guess**.
- `PROMPT_SPOILAGE` — Assess visual spoilage; includes suspension context rule (naturally opaque ≠ cloudiness).
- `PROMPT_BACTERIA` — Bacteria growth estimate with preservative analysis.
- `PROMPT_CHEMICALS` — List chemicals with quantity, category, risk level.

All prompts instruct model to return **valid JSON only**. Regex fallback parser in `engine.py:_extract_json()` handles non-JSON outputs.

**Key:** User text from "Additional Info" box takes priority over OCR text in prompts (engine.py:251-253). OCR text truncated to 800 chars via `format_ocr_for_prompt()` (prompts.py:12).

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

- Model repo is `openbmb/MiniCPM-V-2_6-int4` (**underscore not period** in `2_6`)
- vLLM requires `--quantization bitsandbytes --load-format bitsandbytes` for INT4 (backend.py:67-68)
- Modal endpoint URL via `MODAL_ENDPOINT_URL` env var (`.env` file: `TOKEN` + `MODAL_ENDPOINT_URL`)
- Base64 encoding increases image payload ~33% — JPEG quality 95 preserves fine print
- Modal cold start ~30-60 sec after idle; first request may be slow
- Plotly charts render in `gr.Plot()` — **not** `gr.Image()`
- The Modal backend (`backend.py`) is deployed **separately** from the HF Space
- Image preprocessing enhances contrast + sharpens for medicine labels
- Using `uv` instead of `pip` for faster Modal image builds
- OCR exceptions propagate to error handler instead of failing silently
- Dynamic expiry defaults to 1-year shelf life when dates are missing
- Pass 2 prompts use `ThreadPoolExecutor` for parallel execution (~4x speedup)
- User text priority over OCR text in prompts
- Chemical chart: Name → Quantity with bar width proportional to parsed values
- Modal deployment requires `$env:PYTHONIOENCODING="utf-8"` on Windows
- JSON examples in prompts use escaped braces `{{ }}` to avoid `KeyError`

## Running Locally

```bash
pip install -r requirements.txt
MODAL_ENDPOINT_URL=https://your-workspace--biochem-spoilage-detect-analyze.modal.run python app.py
# Opens at http://localhost:7860
```

## Deploying

### 1. Deploy Modal backend

```bash
pip install modal
$env:PYTHONIOENCODING="utf-8"  # Windows only
modal deploy backend.py
# Prints endpoint URL — save it as MODAL_ENDPOINT_URL secret
```

### 2. Deploy HF Space frontend

1. Set `MODAL_ENDPOINT_URL` as a Space secret (Settings → Secrets)
2. Push code to the HF Space
3. README.md frontmatter already declares `sdk: gradio`, `app_file: app.py`

## Submission Checklist

- [ ] Modal backend deployed and endpoint URL saved
- [ ] HF Space deployed with MODAL_ENDPOINT_URL secret configured
- [ ] Demo video recorded (show: upload image → crop label → analyze → charts + verdict)
- [ ] Social media post made
- [ ] Agent trace published to HF Hub (Sharing is Caring badge)