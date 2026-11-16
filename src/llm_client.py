"""Deepseek R1 client for HF Spaces ZeroGPU orchestration.

Two backend modes:
  1. Remote HTTP endpoint (ZEROGPU_ENDPOINT_URL set): POSTs to a HF Space
     running an OpenAI-compatible /v1/chat/completions API.
  2. Local in-process inference (no URL set): loads
     deepseek-ai/DeepSeek-R1-Distill-Qwen-7B via transformers with 4-bit
     bitsandbytes quantization. Requires GPU + transformers + torch.

The remote mode is used on HF Spaces where the Gradio frontend calls a
separate ZeroGPU Space. The local mode is used for development/testing on
a GPU machine.
"""

from __future__ import annotations

import os
import re
import time

import requests


# Deepseek R1 emits chain-of-thought inside  thinking...</think> blocks.
_THINK_BLOCK = re.compile(r"<think\b[^>]*>(.*?)</think>", re.DOTALL | re.IGNORECASE)
_THINK_TOKEN = re.compile(r"</?think\b[^>]*>", re.IGNORECASE)


def strip_reasoning(text: str) -> str:
    """Remove Deepseek-R1 chain-of-thought from output, keeping the final answer."""
    if not text:
        return ""
    # Drop complete reasoning blocks
    cleaned = _THINK_BLOCK.sub("", text)
    # Drop dangling opening/closing tags (truncated output)
    cleaned = _THINK_TOKEN.sub("", cleaned)
    return cleaned.strip()


class DeepseekR1Client:
    """Client for Deepseek-R1-Distill-Qwen-7B on HF Spaces ZeroGPU.

    Args:
        endpoint_url: Full URL of the ZeroGPU Space chat endpoint. If empty,
            falls back to local in-process model loading.
        model_name: HF repo id for local loading.
        timeout: HTTP timeout for remote calls (ZeroGPU cold start ~30-60s).
    """

    def __init__(
        self,
        endpoint_url: str | None = None,
        model_name: str = "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B",
        timeout: int = 180,
    ) -> None:
        self.endpoint_url = (endpoint_url or os.environ.get("ZEROGPU_ENDPOINT_URL", "")).strip()
        self.model_name = model_name
        self.timeout = timeout
        self._local_model = None
        self._local_tokenizer = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def chat(
        self,
        messages: list[dict],
        temperature: float = 0.2,
        max_tokens: int = 2048,
    ) -> str:
        """Run a chat completion and return the assistant text (reasoning stripped).

        Args:
            messages: OpenAI-style message list [{"role": ..., "content": ...}].
            temperature: Sampling temperature (low for deterministic JSON).
            max_tokens: Max new tokens.
        """
        if self.endpoint_url:
            raw = self._chat_remote(messages, temperature, max_tokens)
        else:
            raw = self._chat_local(messages, temperature, max_tokens)
        return strip_reasoning(raw)

    # ------------------------------------------------------------------
    # Remote mode — HTTP to HF Spaces ZeroGPU (OpenAI-compatible API)
    # ------------------------------------------------------------------

    def _chat_remote(
        self, messages: list[dict], temperature: float, max_tokens: int
    ) -> str:
        url = self._normalize_url(self.endpoint_url)
        payload = {
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }

        last_error: Exception | None = None
        for attempt in range(3):
            try:
                resp = requests.post(url, json=payload, timeout=self.timeout)
                resp.raise_for_status()
                data = resp.json()
                return self._parse_response(data)
            except Exception as e:
                last_error = e
                # ZeroGPU cold start / 503 — back off briefly and retry
                if attempt < 2:
                    time.sleep(3 * (attempt + 1))
        raise RuntimeError(f"Deepseek R1 endpoint failed after 3 attempts: {last_error}")

    @staticmethod
    def _normalize_url(base: str) -> str:
        """Append /v1/chat/completions unless the caller supplied a full path."""
        if base.endswith("/v1/chat/completions"):
            return base
        return base.rstrip("/") + "/v1/chat/completions"

    @staticmethod
    def _parse_response(data: dict) -> str:
        """Extract assistant text from an OpenAI-compatible response."""
        try:
            return data["choices"][0]["message"]["content"] or ""
        except (KeyError, IndexError, TypeError):
            # Some Spaces return {"response": "..."} instead
            if isinstance(data, dict) and "response" in data:
                return str(data["response"])
            return ""

    # ------------------------------------------------------------------
    # Local mode — in-process transformers + bitsandbytes 4-bit
    # ------------------------------------------------------------------

    def _load_local(self) -> None:
        """Lazily load the quantized model on first use."""
        if self._local_model is not None:
            return

        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

        quant_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
        )

        self._local_tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self._local_model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            quantization_config=quant_config,
            device_map="auto",
            torch_dtype=torch.bfloat16,
        )

    def _chat_local(
        self, messages: list[dict], temperature: float, max_tokens: int
    ) -> str:
        self._load_local()
        import torch

        text = self._local_tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = self._local_tokenizer([text], return_tensors="pt").to(
            self._local_model.device
        )

        with torch.no_grad():
            output = self._local_model.generate(
                **inputs,
                max_new_tokens=max_tokens,
                temperature=max(temperature, 0.01),
                do_sample=temperature > 0,
                top_p=0.95,
                pad_token_id=self._local_tokenizer.eos_token_id,
            )

        generated = output[0][inputs["input_ids"].shape[1]:]
        return self._local_tokenizer.decode(generated, skip_special_tokens=True)
