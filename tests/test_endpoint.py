"""Quick test of the HF ZeroGPU VLM endpoint."""
import requests
import json
import base64
import sys
import os
from PIL import Image
import io

url = os.environ.get("VLM_ENDPOINT_URL", "")
if not url:
    print("ERROR: VLM_ENDPOINT_URL is not set. "
          "Deploy the Qwen2.5-VL Space on HF ZeroGPU first.")
    sys.exit(1)

# Create a small test image
img = Image.new("RGB", (100, 100), color=(255, 200, 150))
buf = io.BytesIO()
img.save(buf, format="JPEG", quality=95)
b64 = base64.b64encode(buf.getvalue()).decode()

# HF ZeroGPU endpoint (OpenAI compatible): JSON body with messages and image content
prompt = 'Return ONLY valid JSON: {"status": "ok", "model": "working"}'

print(f"Sending test request to: {url}")
try:
    resp = requests.post(
        f"{url.rstrip('/')}/v1/chat/completions",
        json={
            "messages": [{"role": "user", "content": [{"type": "text", "text": prompt}]}]
        },
        timeout=300,
    )
    print(f"Status: {resp.status_code}")
    if resp.status_code == 200:
        data = resp.json()
        print(f"Response: {data.get('choices', [{}])[0].get('message', {}).get('content', 'NO RESPONSE KEY')[:500]}")
    else:
        print(f"Error body: {resp.text[:500]}")
except Exception as e:
    print(f"Connection error: {e}")