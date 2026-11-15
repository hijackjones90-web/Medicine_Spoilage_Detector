"""HTTP client for Qwen2.5-VL cloud VLM endpoint.

OpenAI-compatible API format (vLLM-served Qwen VL).
Configurable via QWEN_VL_ENDPOINT_URL env var.
"""

import base64
import io
import json
import os
import re

import requests
from PIL import Image


def _image_to_base64(image) -> str:
    """Convert image to base64-encoded JPEG string."""
    if isinstance(image, tuple):
        image = image[0]
    if not isinstance(image, Image.Image):
        try:
            import numpy as np
            image = Image.fromarray(np.array(image))
        except Exception as e:
            raise ValueError(f"Cannot convert image to PIL: {e}")

    if image.mode == "RGBA":
        image = image.convert("RGB")
    elif image.mode != "RGB":
        image = image.convert("RGB")

    max_size = 1344
    if image.width > max_size or image.height > max_size:
        ratio = min(max_size / image.width, max_size / image.height)
        new_size = (int(image.width * ratio), int(image.height * ratio))
        image = image.resize(new_size, Image.Resampling.LANCZOS)

    buf = io.BytesIO()
    image.save(buf, format="JPEG", quality=95)
    return base64.b64encode(buf.getvalue()).decode()


def _extract_json(text: str) -> dict | list | None:
    """Extract JSON from VLM response with multiple fallback strategies."""
    if not text:
        return None

    cleaned = text.strip()

    # Remove markdown code blocks
    for pattern in [r"```json\s*\n?(.*?)\n?\s*```", r"```\s*\n?(.*?)\n?\s*```"]:
        fence_match = re.search(pattern, cleaned, re.DOTALL)
        if fence_match:
            cleaned = fence_match.group(1).strip()
            break

    # Try direct parse
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        pass

    # Find JSON object or array
    for start_char, end_char in [("{", "}"), ("[", "]")]:
        start = cleaned.find(start_char)
        end = cleaned.rfind(end_char)
        if start != -1 and end != -1 and end > start:
            try:
                return json.loads(cleaned[start:end + 1])
            except json.JSONDecodeError:
                pass

    return None


class QwenVLClient:
    """Client for Qwen2.5-VL 7B cloud VLM endpoint.

    Uses OpenAI-compatible API format (vLLM-served Qwen VL).
    Configure via QWEN_VL_ENDPOINT_URL env var.
    """

    def __init__(self, endpoint_url=None, timeout=180, max_retries=3):
        self.endpoint_url = endpoint_url or os.environ.get("QWEN_VL_ENDPOINT_URL")
        if not self.endpoint_url:
            raise RuntimeError(
                "QWEN_VL_ENDPOINT_URL not set. Deploy Qwen VL backend first."
            )
        self.timeout = timeout
        self.max_retries = max_retries

    def _call_endpoint(self, images, prompt, max_tokens=4096, temperature=0.1):
        """Call the Qwen VL endpoint with OpenAI-compatible format."""
        b64_images = [_image_to_base64(img) for img in images[:4]]

        content = [{"type": "text", "text": prompt}]
        for img_b64 in b64_images:
            content.append({
                "type": "image_url",
                "image_url": {"url": f"data:image/jpeg;base64,{img_b64}"},
            })

        payload = {
            "model": os.environ.get("QWEN_VL_MODEL", "Qwen/Qwen2.5-VL-7B-Instruct"),
            "messages": [{"role": "user", "content": content}],
            "max_tokens": max_tokens,
            "temperature": temperature,
        }

        resp = requests.post(
            self.endpoint_url,
            json=payload,
            timeout=self.timeout,
        )
        resp.raise_for_status()
        result = resp.json()
        return result["choices"][0]["message"]["content"]

    def analyze(self, images, prompt):
        """Run a single prompt on images and return raw text."""
        return self._call_endpoint(images, prompt)

    def analyze_with_retry(self, images, prompt):
        """Run prompt with retry logic, returning raw text on success."""
        last_error = None
        for attempt in range(self.max_retries):
            try:
                raw = self._call_endpoint(images, prompt)
                parsed = _extract_json(raw)
                if parsed is not None:
                    return raw
                last_error = "JSON parse failed"
            except Exception as e:
                last_error = str(e)
        raise RuntimeError(
            f"Failed after {self.max_retries} retries. Last error: {last_error}"
        )