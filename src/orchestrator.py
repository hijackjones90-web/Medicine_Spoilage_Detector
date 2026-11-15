"""Agentic orchestrator using Deepseek R1 distill as decision LLM.

Architecture:
  VLM (Qwen2.5-VL) → structured JSON → Deepseek R1 distill (tool-calling)
                                            │
              ┌─────────────────────────────┼─────────────────────────────┐
              ▼                             ▼                             ▼
        score_spoilage              format_report               decide_next_steps
              │                             │                             │
              └─────────────────────────────┼─────────────────────────────┘
                                            ▼
                                 decision dict → AnalysisResult

Fallback: when no DEEPSEEK_ENDPOINT_URL is configured (or the LLM call
fails), the same tools are invoked deterministically in Python, so the
pipeline still produces a verdict. This mirrors the plan's Secondary
("Python deterministic scoring") tier.

The LLM is prompted for a SINGLE tool-call decision per round rather than
a long agentic loop, because 32B-class reasoning models are slow and the
plan targets < 60s end-to-end.
"""

import json
import os

import requests

from src.utils import (
    calculate_spoilage_score,
    get_spoilage_verdict,
    get_verdict_color,
    parse_date,
)

# --- Tool schemas (OpenAI-compatible function calling) ---

TOOL_SCHEMAS = [
    {
        "type": "function",
        "function": {
            "name": "score_spoilage",
            "description": (
                "Calculate the final spoilage score (0-100) and verdict "
                "from visual, bacteria, expiry, color, and dynamic-expiry factors."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "visual_level": {"type": "integer", "minimum": 0, "maximum": 100},
                    "bacteria_level": {"type": "integer", "minimum": 0, "maximum": 100},
                    "days_until_expiry": {"type": "integer"},
                    "shelf_life_days": {"type": "integer"},
                    "color_deviation": {"type": "number", "minimum": 0, "maximum": 1},
                    "dynamic_expiry_days": {"type": "integer"},
                },
                "required": [
                    "visual_level",
                    "bacteria_level",
                    "days_until_expiry",
                    "shelf_life_days",
                ],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "decide_next_steps",
            "description": (
                "Recommend concrete actions based on the verdict, score and "
                "expiry information (e.g. consume, discard, consult pharmacist)."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "verdict": {"type": "string", "enum": ["SAFE", "WARNING", "SPOILED"]},
                    "score": {"type": "integer", "minimum": 0, "maximum": 100},
                },
                "required": ["verdict", "score"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "format_report",
            "description": (
                "Produce a concise human-readable markdown report explaining "
                "the verdict and the key evidence."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "report": {"type": "string"},
                },
                "required": ["report"],
            },
        },
    },
]

DEEPSEEK_ENDPOINT_URL = os.environ.get("DEEPSEEK_ENDPOINT_URL", "")
DEEPSEEK_MODEL = os.environ.get("DEEPSEEK_MODEL", "deepseek-r1-distill-qwen-32b")
DEEPSEEK_API_KEY = os.environ.get("DEEPSEEK_API_KEY", "")
MAX_ROUNDS = int(os.environ.get("ORCHESTRATOR_MAX_ROUNDS", "3"))


# --- Tool implementations (deterministic Python fallbacks) ---


def tool_score_spoilage(arguments: dict) -> dict:
    """score_spoilage tool: deterministic spoilage scoring."""
    score = calculate_spoilage_score(
        visual_level=int(arguments.get("visual_level", 50) or 0),
        bacteria_level=int(arguments.get("bacteria_level", 0) or 0),
        days_until_expiry=int(arguments.get("days_until_expiry", 365) or 365),
        shelf_life_days=int(arguments.get("shelf_life_days", 365) or 365),
        color_deviation=float(arguments.get("color_deviation", 0.0) or 0.0),
        dynamic_expiry_days=arguments.get("dynamic_expiry_days"),
    )
    return {
        "score": score,
        "verdict": get_spoilage_verdict(score),
        "verdict_color": get_verdict_color(score),
    }


def tool_decide_next_steps(arguments: dict) -> dict:
    """decide_next_steps tool: rule-based action recommendations."""
    verdict = arguments.get("verdict", "WARNING")
    score = arguments.get("score", 50)

    if verdict == "SPOILED":
        actions = [
            "Do not consume this medicine.",
            "Dispose of it safely (do not flush; follow local medicine take-back rules).",
            "Consult a pharmacist for a replacement.",
            "Check whether the batch has been recalled.",
        ]
        urgency = "high"
    elif verdict == "WARNING":
        actions = [
            "Use with caution and only after a pharmacist check.",
            "Look for further changes (color, smell, sediment) before each dose.",
            "Prefer using it within the next 30 days.",
        ]
        urgency = "medium"
    else:
        actions = [
            "Safe to use as directed.",
            "Store in a cool, dry place away from light.",
            "Re-check the expiry date before use.",
        ]
        urgency = "low"

    return {"verdict": verdict, "score": score, "urgency": urgency, "actions": actions}


def tool_format_report(arguments: dict) -> dict:
    """format_report tool: coerce/accept a markdown report string."""
    report = arguments.get("report", "")
    if not isinstance(report, str):
        report = str(report)
    return {"report": report}


TOOL_REGISTRY = {
    "score_spoilage": tool_score_spoilage,
    "decide_next_steps": tool_decide_next_steps,
    "format_report": tool_format_report,
}


def _build_context(result) -> dict:
    """Build a compact decision context from an AnalysisResult-like object."""
    medicine_info = getattr(result, "medicine_info", {}) or {}
    spoilage = getattr(result, "spoilage_assessment", {}) or {}
    bacteria = getattr(result, "bacteria_estimate", {}) or {}
    chemicals = getattr(result, "chemicals", []) or []
    dynamic_expiry = getattr(result, "dynamic_expiry", {}) or {}
    color_analysis = getattr(result, "color_analysis", {}) or {}

    mfg = parse_date(medicine_info.get("mfg_date"))
    exp = parse_date(medicine_info.get("exp_date"))
    shelf_life_days = 365
    days_until_expiry = 365
    if mfg and exp:
        shelf_life_days = max(30, (exp - mfg).days)
    if exp:
        from datetime import datetime
        days_until_expiry = (exp - datetime.now()).days

    return {
        "medicine_info": medicine_info,
        "spoilage_assessment": spoilage,
        "bacteria_estimate": bacteria,
        "chemicals": chemicals,
        "dynamic_expiry": dynamic_expiry,
        "color_analysis": color_analysis,
        "derived": {
            "visual_level": spoilage.get("spoilage_level", 50),
            "bacteria_level": bacteria.get("growth_level", 0),
            "days_until_expiry": days_until_expiry,
            "shelf_life_days": shelf_life_days,
            "color_deviation": color_analysis.get("color_deviation", 0.0),
            "dynamic_expiry_days": dynamic_expiry.get("days_until_dynamic_expiry"),
        },
    }


def _system_prompt() -> str:
    return (
        "You are a drug spoilage decision orchestrator. You receive structured "
        "vision-model output about a medicine package and must decide spoilage.\n"
        "Rules:\n"
        "1. Call score_spoilage first using the provided derived factors.\n"
        "2. Then call decide_next_steps with the returned verdict and score.\n"
        "3. Finally call format_report with a concise markdown report.\n"
        "Use the provided values; do not invent data. Call tools one at a time."
    )


def _llm_call(messages: list) -> dict:
    """Call the Deepseek R1 distill endpoint (OpenAI-compatible)."""
    headers = {"Content-Type": "application/json"}
    if DEEPSEEK_API_KEY:
        headers["Authorization"] = f"Bearer {DEEPSEEK_API_KEY}"

    payload = {
        "model": DEEPSEEK_MODEL,
        "messages": messages,
        "tools": TOOL_SCHEMAS,
        "tool_choice": "auto",
        "temperature": 0.2,
        "max_tokens": 2048,
    }

    resp = requests.post(
        DEEPSEEK_ENDPOINT_URL,
        headers=headers,
        json=payload,
        timeout=180,
    )
    resp.raise_for_status()
    return resp.json()


def _run_llm_loop(context: dict) -> dict:
    """Run the tool-calling loop with Deepseek R1 until a decision is formed."""
    messages = [
        {"role": "system", "content": _system_prompt()},
        {"role": "user", "content": json.dumps(context, default=str, indent=2)},
    ]

    decision = {}
    for _ in range(MAX_ROUNDS):
        response = _llm_call(messages)
        message = response["choices"][0]["message"]
        tool_calls = message.get("tool_calls") or []

        if not tool_calls:
            # Model answered in plain text — treat as report if we have none
            if not decision.get("report") and message.get("content"):
                decision["report"] = message["content"]
            break

        messages.append(message)
        for call in tool_calls:
            fn_name = call["function"]["name"]
            try:
                args = json.loads(call["function"].get("arguments") or "{}")
            except json.JSONDecodeError:
                args = {}

            tool = TOOL_REGISTRY.get(fn_name)
            if not tool:
                tool_result = {"error": f"unknown tool: {fn_name}"}
            else:
                try:
                    tool_result = tool(args)
                except Exception as e:
                    tool_result = {"error": str(e)}

            # Merge tool outputs into decision
            if fn_name == "score_spoilage":
                decision.update(tool_result)
            elif fn_name == "decide_next_steps":
                decision["next_steps"] = tool_result
            elif fn_name == "format_report":
                decision["report"] = tool_result.get("report", "")

            messages.append({
                "role": "tool",
                "tool_call_id": call.get("id", fn_name),
                "content": json.dumps(tool_result, default=str),
            })

        if decision.get("report") and decision.get("verdict"):
            break

    decision["source"] = "llm"
    return decision


def _run_python_fallback(context: dict) -> dict:
    """Deterministic fallback: call the same tools in Python."""
    derived = context["derived"]
    score_result = tool_score_spoilage(derived)
    next_steps = tool_decide_next_steps({
        "verdict": score_result["verdict"],
        "score": score_result["score"],
    })

    info = context.get("medicine_info", {})
    name = info.get("name", "this medicine")
    report = (
        f"**{name}** — spoilage score **{score_result['score']}/100** "
        f"(**{score_result['verdict']}**).\n\n"
        f"Visual level: {derived['visual_level']}/100 · "
        f"Bacteria level: {derived['bacteria_level']}/100 · "
        f"Days until expiry: {derived['days_until_expiry']}."
    )

    return {
        "score": score_result["score"],
        "verdict": score_result["verdict"],
        "verdict_color": score_result["verdict_color"],
        "next_steps": next_steps,
        "report": report,
        "source": "python_fallback",
    }


def orchestrate(result, user_text: str = "") -> dict:
    """Produce a spoilage decision from an AnalysisResult.

    Uses Deepseek R1 tool-calling when DEEPSEEK_ENDPOINT_URL is configured;
    otherwise falls back to deterministic Python scoring.

    Returns a decision dict with keys: score, verdict, verdict_color,
    next_steps, report, source.
    """
    context = _build_context(result)
    if user_text and user_text.strip():
        context["user_text"] = user_text.strip()

    if not DEEPSEEK_ENDPOINT_URL:
        return _run_python_fallback(context)

    try:
        decision = _run_llm_loop(context)
        # Ensure the decision is complete; fill gaps deterministically.
        if "score" not in decision or "verdict" not in decision:
            fallback = _run_python_fallback(context)
            fallback.update({k: v for k, v in decision.items() if v})
            return fallback
        return decision
    except Exception as e:
        fallback = _run_python_fallback(context)
        fallback["llm_error"] = str(e)
        return fallback


def apply_decision(result, decision: dict) -> None:
    """Attach the orchestrator decision to an AnalysisResult's raw responses."""
    result.raw_responses["orchestrator_decision"] = json.dumps(
        decision, default=str, indent=2
    )
    result.raw_responses["orchestrator_source"] = decision.get("source", "unknown")