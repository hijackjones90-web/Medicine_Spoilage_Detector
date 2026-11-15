# Orchestration Plan — Qwen VL + Deepseek R1 Distill

**Goal:** Replace current deterministic pipeline with agentic orchestration using Qwen VL as vision model and Deepseek R1 distill as decision orchestrator.

**Saved at:** `/workspace/app/.kilo/plans/1751264110195-qwen-deepseek-r1-orchestration-plan.md`

---

## Model Choices

### VLM — Qwen2.5-VL 7B (6.0 GB download, Q4_K_M)

| Property | Value |
|---|---|
| Params | 7B (Qwen2.5-VL base) |
| Download size | ~6.0 GB (Q4) |
| VRAM (4-bit) | ~6–8 GB |
| Context | 125K tokens |
| Strengths | OCR, document understanding, JSON output, charts reading |
| Ollama command | `ollama pull qwen2.5vl:7b` |
| Use case | Cloud GPU deployment (Kaggle/Colab P100/T4) or local 8GB+ GPU |

### Decision LLM — Deepseek-R1-Distill-Qwen-2.5-32B (4-bit, ~20 GB)

| Property | Value |
|---|---|
| Base | Qwen2.5-32B |
| Distilled from | Deepseek R1 |
| VRAM (4-bit) | ~20 GB (large, needs 24+ GB GPU or Apple Silicon 32 GB unified) |
| Context | 128K tokens |
| Strengths | Reasoning, structured output, tool-use patterns, chain-of-thought |
| Ollama command | Not in standard Ollama library — use vLLM or llama.cpp |
| Use case | Cloud GPU deployment (HTTPS endpoint) |

### Orchestration Architecture

```
User Upload → Gradio UI → Cloud VLM (Qwen2.5-VL 7B) → JSON output
                               │
                               ▼
                    Decision LLM (Deepseek R1 distill) ← Tool calls
                               │
          ────────────────────────────────────────────────
          │                 │                 │
  score_spoilage    format_report   decide_next_steps   fallback_Python
          │                 │                 │
          ▼                 ▼                 ▼
    Scored verdict → Markdown report → UI display
```

### Communication Protocol

**VLM output → LLM input (structured JSON):**
```json
{
  "medicine_info": {
    "name": "...",
    "mfg_date": "DD/MM/YYYY or null",
    "exp_date": "DD/MM/YYYY or null",
    "batch_no": "...",
    "dosage_form": "...",
    "ingredients": ["..."],
    "storage_conditions": "..."
  },
  "spoilage_assessment": {
    "discoloration": true/false,
    "cloudiness": true/false,
    "sediment": true/false,
    "seal_intact": true/false,
    "label_damage": true/false,
    "spoilage_level": 0-100,
    "indicators": ["..."]
  },
  "bacteria_estimate": {
    "growth_level": 0-100,
    "preservatives_found": ["..."],
    "reasoning": "..."
  },
  "chemicals": [
    {"name": "...", "quantity": "...", "category": "...", "risk_level": "..."}
  ]
}
```

**LLM → Tools (function calling):**
```json
{
  "name": "score_spoilage",
  "arguments": {
    "visual_level": 45,
    "bacteria_level": 30,
    "days_until_expiry": -5,
    "shelf_life_days": 548,
    "color_deviation": 0.25,
    "dynamic_expiry_days": 30
  }
}
```

**Report format (LLM output → Gradio):**
- JSON: structured verdict + scores
- Markdown: human-readable explanation + reasoning

### Tool Definitions (local LLM calls)

| Tool | Purpose | Inputs | Output |
|---|---|---|---|
| `score_spoilage` | Calculate spoilage score + verdict | visual_level, bacteria_level, days_until_expiry, shelf_life_days, color_deviation, dynamic_expiry_days | score (0-100), verdict (SAFE/WARNING/SPOILED) |
| `format_report` | Generate human-readable report | All VLM outputs + score + verdict | markdown string for Gradio |
| `decide_next_steps` | Recommend actions based on verdict | verdict, score, expiry info | action list + advice text |

### Fallback Strategy

1. **Primary:** VLM cloud (Kaggle GPU) → LLM local (Deepseek R1 distill via vLLM)
2. **Secondary:** VLM cloud → Python deterministic scoring (existing `utils.py`)
3. **Tertiary:** Local VLM fallback (Moondream 1.8B or Qwen2.5-VL 3B on CPU) + Python scoring

### Model Deployment

| Component | Runtime | Location | Notes |
|---|---|---|---|
| Qwen2.5-VL 7B | vLLM server (bitsandbytes INT4) | Cloud GPU (Kaggle/Colab) | HTTPS endpoint |
| Deepseek R1 distill Qwen2.5-32B | vLLM server (4-bit) | Cloud GPU (same or separate) | Tool-calling endpoint |
| Python fallback | Native (no GPU needed) | Anywhere | `utils.py` functions |

### Implementation Steps

1. **Infrastructure:**
   - Kaggle account with phone verification (30 GPU-hrs/week)
   - Deploy Qwen2.5-VL 7B on Kaggle/Colab via vLLM
   - Deploy Deepseek R1 distill Qwen2.5-32B on cloud GPU (separate container or same with different model)
   - Install local vLLM for Deepseek R1 distill (if running locally on laptop)

2. **Code changes:**
   - Modify `app.py` to use orchestrator workflow instead of direct `analyze_image`
   - Create `orchestrator.py`: tool-calling LLM with score_spoilage, format_report, decide_next_steps
   - Create `cloud_vlm_client.py`: VLM inference on cloud GPU (Qwen2.5-VL 7B)
   - Update `engine.py` to support both cloud and local VLM modes

3. **Tool definitions:**
   - Define function schemas for score_spoilage, format_report, decide_next_steps
   - Parse LLM tool calls and execute corresponding Python functions
   - Aggregate results into Gradio outputs

4. **Testing:**
   - End-to-end: upload image → VLM JSON → LLM tool calls → Gradio report
   - Fallback test: kill cloud, use Python scoring
   - Validate report format matches current output structure

5. **Deployment:**
   - HF Space with local orchestration logic
   - Cloud GPU endpoints as VLM backends (Kaggle)
   - Keep deterministic Python scoring as fallback

### Risks & Constraints

| Risk | Mitigation |
|---|---|
| Cloud GPU quota (30 hrs/week) | Batch process images; schedule heavy usage off-peak |
| Deepseek R1 distill VRAM (20+ GB) | Use cloud GPU only; do not run locally on 8GB laptop |
| LLM tool-call latency (sequential) | Precompute VLM JSON first, then LLM — ~30s total |
| Model drift / output format changes | Keep Python scoring fallback; validate JSON schemas |
| Qwen2.5-VL 7B OCR accuracy on medicine labels | Same prompts as current `engine.py`; validate with test images |

### Validation Plan

1. **Functional:** End-to-end test with sample medicine image
   - Upload → VLM extracts all fields → LLM scores → Gradio displays correctly
   - Compare LLM verdict vs current Python scoring accuracy

2. **Structural:** Verify all 11 Gradio outputs are populated
   - info_md, verdict_md, date_md, chemical_chart, timeline_chart, gauge_chart, growth_curve_chart, color_timeline_chart, dynamic_expiry_chart, radar_chart, raw_md

3. **Fallback:** Test kill-switch disables cloud, Python scoring takes over
   - No regressions in Gradio UI

4. **Performance:** Measure total latency (VLM + LLM)
   - Target: <60s for typical image (cloud GPU + local LLM)
   - Acceptable: <120s if cloud GPU under load

### Out of Scope

- Running Deepseek R1 distill locally on CPU-only laptop (20+ GB VRAM required)
- Training or fine-tuning either model
- Multi-modal reasoning beyond structured tool calling
- Persistent cloud GPU deployment beyond hackathon submission window

### Open Questions (if any)

- Should Deepseek R1 distill run on same cloud GPU as Qwen VL (separate containers) or different endpoint?
- What exact function schema format does the local LLM tool-calling framework expect?
- Should `format_report` generate markdown directly or return structured data for Gradio to format?

---
**Next:** Proceed with infrastructure setup (Kaggle account, vLLM deployment) and orchestrator code prototype.