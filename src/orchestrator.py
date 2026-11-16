"""Orchestration layer for Drug Spoilage Detector.

Architecture:
  VLM (Qwen2.5-VL-7B on Modal) — OCR extraction + structured analysis (4 parallel prompts)
  Orchestrator (Deepseek-R1-Distill-Qwen-7B on HF Spaces ZeroGPU) — single synthesis call
  Python fallbacks (utils.py) — bacteria growth, color analysis, dynamic expiry

Design principles (from plan):
  - Precompute VLM JSON first, then a single LLM call for verdict + report.
  - No multi-step tool-calling loops — too slow on 7B-class orchestrators.
  - Preserve AnalysisResult contract so extract_drug_outputs() works unchanged.
"""

from __future__ import annotations

import base64
import io
import json
import os
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from datetime import datetime, timedelta

import requests
from PIL import Image

from src.sagemaker_client import (
    _image_to_base64,
    _preprocess_images_for_medicine,
    _extract_json,
    _run_prompt as _vlm_run_prompt,
)
from src.prompts import PROMPT_OCR, PASS2_PROMPTS, format_ocr_for_prompt
from src.utils import (
    parse_date,
    calculate_spoilage_score,
    get_spoilage_verdict,
    get_verdict_color,
    calculate_theoretical_growth,
    calculate_dynamic_expiry,
    estimate_color_from_spoilage,
    enrich_chemicals,
)
from src.llm_client import DeepseekR1Client


MAX_RETRIES = 3


@dataclass
class AnalysisResult:
    """Structured result from orchestrator analysis — same fields as engine.AnalysisResult."""

    ocr_data: dict = field(default_factory=dict)
    medicine_info: dict = field(default_factory=dict)
    spoilage_assessment: dict = field(default_factory=dict)
    bacteria_estimate: dict = field(default_factory=dict)
    chemicals: list = field(default_factory=list)
    raw_responses: dict = field(default_factory=dict)
    errors: list = field(default_factory=list)
    bacteria_growth_curve: dict = field(default_factory=dict)
    color_analysis: dict = field(default_factory=dict)
    dynamic_expiry: dict = field(default_factory=dict)


class Orchestrator:
    """Coordinates VLM extraction + Deepseek R1 synthesis for spoilage detection.

    Replaces engine.analyze_image() while preserving the AnalysisResult contract
    used by app.extract_drug_outputs().
    """

    def __init__(self) -> None:
        self.zerogpu_url = os.environ.get("ZEROGPU_ENDPOINT_URL", "")
        self._deepseek: DeepseekR1Client | None = None

    @property
    def deepseek(self) -> DeepseekR1Client:
        """Lazy-initialized Deepseek R1 client."""
        if self._deepseek is None:
            self._deepseek = DeepseekR1Client(endpoint_url=self.zerogpu_url)
        return self._deepseek

    # ------------------------------------------------------------------
    # Public API — drop-in replacement for engine.analyze_image()
    # ------------------------------------------------------------------

    def analyze(self, images: list, user_text: str = "") -> AnalysisResult:
        """Run the full orchestration pipeline.

        1. VLM Pass 1 — OCR extraction
        2. VLM Pass 2 — 4 parallel structured prompts
        3. Deepseek R1 — single synthesis call (verdict + reasoning)
        4. Python fallbacks — growth curve, color analysis, dynamic expiry

        Returns an AnalysisResult compatible with extract_drug_outputs().
        """
        result = AnalysisResult()

        if not images:
            result.errors.append("No images provided")
            return result

        # Preprocess images for medicine label analysis
        try:
            processed_images = _preprocess_images_for_medicine(images)
        except Exception as e:
            result.errors.append(f"Image preprocessing failed: {str(e)}")
            processed_images = images

        # --- Step 1: VLM Pass 1 — OCR extraction ---
        try:
            ocr_data = self._pass1_ocr(processed_images)
            result.ocr_data = ocr_data
            result.raw_responses["ocr"] = json.dumps(ocr_data, indent=2)
        except Exception as e:
            result.errors.append(f"OCR failed: {str(e)}")
            ocr_data = {"text_blocks": [], "packaging_type": "unknown", "surfaces_visible": []}

        if user_text and user_text.strip():
            result.raw_responses["user_text"] = user_text.strip()

        # --- Step 2: VLM Pass 2 — 4 parallel structured prompts ---
        try:
            structured = self._pass2_structured(processed_images, ocr_data, user_text=user_text)

            if "info" in structured and "_error" not in structured["info"]:
                result.medicine_info = structured["info"]
                result.raw_responses["vlm_info"] = json.dumps(structured["info"], indent=2)
            else:
                result.errors.append(
                    f"info: {structured.get('info', {}).get('_error', 'failed')}"
                )

            if "spoilage" in structured and "_error" not in structured["spoilage"]:
                result.spoilage_assessment = structured["spoilage"]
                result.raw_responses["vlm_spoilage"] = json.dumps(
                    structured["spoilage"], indent=2
                )
            else:
                result.errors.append(
                    f"spoilage: {structured.get('spoilage', {}).get('_error', 'failed')}"
                )

            if "bacteria" in structured and "_error" not in structured["bacteria"]:
                result.bacteria_estimate = structured["bacteria"]
                result.raw_responses["vlm_bacteria"] = json.dumps(
                    structured["bacteria"], indent=2
                )
            else:
                result.errors.append(
                    f"bacteria: {structured.get('bacteria', {}).get('_error', 'failed')}"
                )

            if "chemicals" in structured and "_error" not in structured["chemicals"]:
                chem_data = structured["chemicals"]
                if isinstance(chem_data, dict) and "chemicals" in chem_data:
                    result.chemicals = chem_data["chemicals"]
                elif isinstance(chem_data, list):
                    result.chemicals = chem_data
                result.raw_responses["vlm_chemicals"] = json.dumps(chem_data, indent=2)
            else:
                result.errors.append(
                    f"chemicals: {structured.get('chemicals', {}).get('_error', 'failed')}"
                )
        except Exception as e:
            result.errors.append(f"VLM Pass 2 failed: {str(e)}")

        # --- Step 3: Deepseek R1 — single synthesis call ---
        try:
            synthesis = self._synthesize_with_deepseek(
                images=processed_images,
                ocr_data=result.ocr_data,
                medicine_info=result.medicine_info,
                spoilage_assessment=result.spoilage_assessment,
                bacteria_estimate=result.bacteria_estimate,
                chemicals=result.chemicals,
                user_text=user_text,
            )
            result.raw_responses["deepseek_synthesis"] = json.dumps(synthesis, indent=2)

            # Allow Deepseek to override/refine VLM extractions
            if synthesis.get("medicine_info") and isinstance(synthesis["medicine_info"], dict):
                result.medicine_info.update(synthesis["medicine_info"])
            if synthesis.get("spoilage_assessment") and isinstance(synthesis["spoilage_assessment"], dict):
                result.spoilage_assessment.update(synthesis["spoilage_assessment"])
            if synthesis.get("bacteria_estimate") and isinstance(synthesis["bacteria_estimate"], dict):
                result.bacteria_estimate.update(synthesis["bacteria_estimate"])
            if synthesis.get("verdict_reasoning"):
                result.raw_responses["deepseek_reasoning"] = synthesis["verdict_reasoning"]
        except Exception as e:
            result.errors.append(f"Deepseek R1 synthesis failed: {str(e)}")

        # --- Step 4: Python fallback calculations ---
        self._python_fallbacks(result)

        return result

    # ------------------------------------------------------------------
    # VLM pipeline (same as engine, delegated to Modal/Qwen2.5-VL)
    # ------------------------------------------------------------------

    def _run_prompt_with_retry(self, images: list, prompt: str) -> str:
        """Run a VLM prompt with retry-on-failure and prompt simplification."""
        last_error = None
        for attempt in range(MAX_RETRIES):
            try:
                current_prompt = self._simplify_prompt(prompt, attempt)
                raw = _vlm_run_prompt(images, current_prompt)
                parsed = _extract_json(raw)
                if parsed is not None:
                    return raw
                last_error = f"JSON parse failed on attempt {attempt + 1}"
            except Exception as e:
                last_error = str(e)
        raise RuntimeError(f"Failed after {MAX_RETRIES} retries. Last error: {last_error}")

    @staticmethod
    def _simplify_prompt(prompt: str, retry_num: int) -> str:
        """Progressively simplify prompt on each retry (mirrors engine._simplify_prompt)."""
        if retry_num == 0:
            return prompt
        elif retry_num == 1:
            lines = [
                l for l in prompt.split("\n")
                if not l.strip().startswith("Note:") and "example" not in l.lower()
            ]
            return "\n".join(lines)
        elif retry_num == 2:
            json_start = prompt.find("{")
            json_end = prompt.rfind("}") + 1
            if json_start != -1 and json_end > json_start:
                return f"Return ONLY valid JSON matching this schema:\n{prompt[json_start:json_end]}"
            return prompt
        elif retry_num == 3:
            first_line = prompt.split("\n")[0]
            json_start = prompt.find("{")
            json_end = prompt.rfind("}") + 1
            if json_start != -1:
                return f"{first_line}\nReturn JSON: {prompt[json_start:json_end]}"
            return first_line
        return "Return valid JSON only. No explanation."

    def _pass1_ocr(self, images: list) -> dict:
        """Pass 1: Extract visible text from all images."""
        raw = self._run_prompt_with_retry(images, PROMPT_OCR)
        parsed = _extract_json(raw)
        if isinstance(parsed, dict):
            return parsed
        return {"text_blocks": [], "packaging_type": "unknown", "surfaces_visible": []}

    def _pass2_structured(
        self, images: list, ocr_data: dict, user_text: str = ""
    ) -> dict[str, dict | list]:
        """Pass 2: Run 4 VLM prompts IN PARALLEL with OCR context."""
        ocr_text = format_ocr_for_prompt(ocr_data, max_chars=800)

        user_context = ""
        if user_text and user_text.strip():
            user_context = f"User-provided info:\n{user_text.strip()}\n\n"

        prompts_to_run = {}
        for prompt_name, prompt_template in PASS2_PROMPTS.items():
            if "{OCR_TEXT}" in prompt_template:
                combined = user_context + "OCR text:\n" + ocr_text if user_context else ocr_text
                prompts_to_run[prompt_name] = prompt_template.format(OCR_TEXT=combined)
            else:
                prompts_to_run[prompt_name] = prompt_template

        results: dict[str, dict | list] = {}
        with ThreadPoolExecutor(max_workers=4) as executor:
            futures = {
                executor.submit(self._run_prompt_with_retry, images, prompt): name
                for name, prompt in prompts_to_run.items()
            }
            for future in as_completed(futures):
                name = futures[future]
                try:
                    raw = future.result()
                    parsed = _extract_json(raw)
                    if parsed is not None:
                        results[name] = parsed
                    else:
                        results[name] = {"_error": "JSON parse failed after retries"}
                except Exception as e:
                    results[name] = {"_error": str(e)}

        return results

    # ------------------------------------------------------------------
    # Deepseek R1 synthesis — single call, no multi-step tool loops
    # ------------------------------------------------------------------

    def _synthesize_with_deepseek(
        self,
        images: list,
        ocr_data: dict,
        medicine_info: dict,
        spoilage_assessment: dict,
        bacteria_estimate: dict,
        chemicals: list,
        user_text: str = "",
    ) -> dict:
        """Send all VLM results to Deepseek R1 for final synthesis.

        Single call — returns verdict, reasoning, and any corrections to VLM extractions.
        """
        ocr_text = format_ocr_for_prompt(ocr_data, max_chars=800)

        synthesis_prompt = f"""You are a pharmaceutical quality assurance expert reviewing a VLM analysis of medicine packaging.

VLM EXTRACTED DATA:
- Medicine info: {json.dumps(medicine_info, indent=2)[:1200]}
- Spoilage assessment: {json.dumps(spoilage_assessment, indent=2)[:800]}
- Bacteria estimate: {json.dumps(bacteria_estimate, indent=2)[:800]}
- Chemicals: {json.dumps(chemicals, indent=2)[:1000]}
- OCR text: {ocr_text[:800]}
- User info: {user_text[:500] if user_text else "none"}

TASK: Review the VLM extraction for errors and inconsistencies. Then return a final assessment.

Return ONLY valid JSON:
{{
  "medicine_info": {{"name": "...", "manufacturer": "...", "mfg_date": "...", "exp_date": "...", "batch_no": "...", "dosage_form": "...", "ingredients": [...], "storage_conditions": "..."}},
  "spoilage_assessment": {{"discoloration": true/false, "cloudiness": true/false, "sediment": true/false, "seal_intact": true/false, "label_damage": true/false, "spoilage_level": 0-100, "indicators": [...]}},
  "bacteria_estimate": {{"growth_level": 0-100, "preservatives_found": [...], "reasoning": "..."}},
  "verdict": "SAFE|WARNING|SPOILED",
  "verdict_reasoning": "brief explanation of final verdict",
  "confidence": 0-100,
  "corrections": ["list of any VLM errors corrected"]
}}

Rules:
- If VLM spoilage_level > 60 but seal is intact and no visual indicators, reduce spoilage_level.
- Suspensions (iron syrup etc.) are naturally opaque — do not flag as cloudiness.
- Herbal/Ayurvedic ingredients are low risk unless visibly spoiled.
- Set dates to null if not clearly visible — do not guess.
- Confidence should reflect data quality, not optimism.
"""

        # Build image content for Deepseek (if it supports vision) — text-only fallback
        response = self.deepseek.chat(
            messages=[
                {
                    "role": "user",
                    "content": synthesis_prompt,
                }
            ],
            temperature=0.2,
            max_tokens=2048,
        )

        parsed = _extract_json(response)
        if parsed is not None and isinstance(parsed, dict):
            return parsed

        # Fallback: return minimal structure so pipeline continues
        return {
            "medicine_info": medicine_info,
            "spoilage_assessment": spoilage_assessment,
            "bacteria_estimate": bacteria_estimate,
            "verdict": "UNKNOWN",
            "verdict_reasoning": "Deepseek synthesis failed — using VLM data directly",
            "confidence": 0,
            "corrections": [],
        }

    # ------------------------------------------------------------------
    # Python fallback calculations (delegated to utils.py)
    # ------------------------------------------------------------------

    def _python_fallbacks(self, result: AnalysisResult) -> None:
        """Compute bacteria growth curve, color analysis, and dynamic expiry."""

        # Bacteria growth curve
        try:
            ingredients = result.medicine_info.get("ingredients", [])
            preservatives = result.bacteria_estimate.get("preservatives_found", [])
            spoilage_level = result.spoilage_assessment.get("spoilage_level", 0)
            vlm_bacteria_level = result.bacteria_estimate.get("growth_level", 0)
            days_since_mfg = self._days_since_mfg(result.medicine_info)
            shelf_life = self._shelf_life(result.medicine_info)

            result.bacteria_growth_curve = calculate_theoretical_growth(
                ingredients=ingredients,
                preservatives=preservatives,
                shelf_life_days=shelf_life,
                days_since_mfg=days_since_mfg,
                spoilage_level=spoilage_level,
                vlm_bacteria_level=vlm_bacteria_level,
            )
            result.raw_responses["python_bacteria_growth"] = json.dumps(
                result.bacteria_growth_curve, indent=2
            )
        except Exception as e:
            result.errors.append(f"Bacteria growth calc: {str(e)}")

        # Color analysis
        try:
            result.color_analysis = estimate_color_from_spoilage(result.spoilage_assessment)
            result.raw_responses["python_color_analysis"] = json.dumps(
                result.color_analysis, indent=2
            )
        except Exception as e:
            result.errors.append(f"Color analysis calc: {str(e)}")

        # Dynamic expiry
        try:
            mfg_date = parse_date(result.medicine_info.get("mfg_date"))
            exp_date = parse_date(result.medicine_info.get("exp_date"))
            preservatives = result.bacteria_estimate.get("preservatives_found", [])
            color_deviation = result.color_analysis.get("color_deviation", 0.0)

            result.dynamic_expiry = calculate_dynamic_expiry(
                mfg_date=mfg_date,
                exp_date=exp_date,
                spoilage_assessment=result.spoilage_assessment,
                color_deviation=color_deviation,
                preservatives=preservatives,
            )
            result.raw_responses["python_dynamic_expiry"] = json.dumps(
                result.dynamic_expiry, indent=2
            )
        except Exception as e:
            result.errors.append(f"Dynamic expiry calc: {str(e)}")

    @staticmethod
    def _days_since_mfg(medicine_info: dict) -> int:
        mfg_date_str = medicine_info.get("mfg_date")
        if mfg_date_str:
            mfg_date = parse_date(mfg_date_str)
            if mfg_date:
                return (datetime.now() - mfg_date).days
        return 60

    @staticmethod
    def _shelf_life(medicine_info: dict) -> int:
        mfg_date_str = medicine_info.get("mfg_date")
        exp_date_str = medicine_info.get("exp_date")
        if mfg_date_str and exp_date_str:
            mfg_date = parse_date(mfg_date_str)
            exp_date = parse_date(exp_date_str)
            if mfg_date and exp_date:
                return max(30, (exp_date - mfg_date).days)
        return 365


# ------------------------------------------------------------------
# Convenience function — drop-in replacement for engine.analyze_image()
# ------------------------------------------------------------------

def analyze_image(images: list, user_text: str = "") -> AnalysisResult:
    """Analyze medicine images via VLM + Deepseek R1 orchestration.

    Drop-in replacement for src.engine.analyze_image().
    Returns AnalysisResult compatible with app.extract_drug_outputs().
    """
    orch = Orchestrator()
    return orch.analyze(images, user_text=user_text)
