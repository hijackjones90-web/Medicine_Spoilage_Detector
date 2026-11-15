"""Quick test of the VLM endpoint (Kaggle GPU or Ollama)."""
import requests
import json
import base64
import sys
from PIL import Image
import io

# Try to read endpoint URL from environment
# Priority: KAGGLE_VLM_URL > OLLAMA_HOST > default test endpoint
import os

KAGGLE_VLM_URL = os.environ.get("KAGGLE_VLM_URL", "")
OLLAMA_HOST = os.environ.get("OLLAMA_HOST", "http://localhost:11434")

# Determine endpoint URL
if KAGGLE_VLM_URL:
    url = KAGGLE_VLM_URL
else:
    # Try Ollama first
    try:
        # Test Ollama endpoint
        test_payload = json.dumps({
            "model": "qwen2.5vl:7b",
            "messages": [{"role": "user", "content": [{"type": "text", "text": "test"}]}],
            "stream": False
        }).encode()
        
        resp = requests.post(
            f"{OLLAMA_HOST}/api/generate",
            data=test_payload,
            headers={"Content-Type": "application/json"},
            timeout=5,
        )
        if resp.status_code == 200:
            url = f"{OLLAMA_HOST}/api/generate"
            print("Using Ollama endpoint")
        else:
            url = "https://api.kaggle.com/v1/notebooks/..."  # placeholder
    except:
        url = ""

# If no endpoint found, exit
if not url:
    print("No VLM endpoint configured. Set KAGGLE_VLM_URL or OLLAMA_HOST env var.")
    sys.exit(1)

# Create a small test image
img = Image.new("RGB", (100, 100), color=(255, 200, 150))
buf = io.BytesIO()
img.save(buf, format="JPEG", quality=95)
b64 = base64.b64encode(buf.getvalue()).decode()

print(f"Testing VLM endpoint: {url}")

if url.startswith("http://localhost") or url.startswith("http://127.0.0.1"):
    # Ollama endpoint
    prompt = 'Return ONLY valid JSON: {"status": "ok", "model": "working"}'
    
    try:
        resp = requests.post(
            f"{url}",
            json={
                "model": "qwen2.5vl:7b",
                "messages": [{"role": "user", "content": [{"type": "text", "text": prompt}]}]
            },
            timeout=120,
        )
        print(f"Status: {resp.status_code}")
        if resp.status_code == 200:
            data = resp.json()
            print(f"Response: {data.get('message', {}).get('content', 'NO RESPONSE')[:500]}")
        else:
            print(f"Error: {resp.text[:500]}")
    except Exception as e:
        print(f"Connection error: {e}")
elif "kaggle.com" in url:
    # Kaggle GPU endpoint (vLLM style)
    prompt = 'Return ONLY valid JSON: {"status": "ok", "model": "working"}'
    
    try:
        resp = requests.post(
            url,
            params={"prompt": prompt},
            json=[b64],
            timeout=120,
        )
        print(f"Status: {resp.status_code}")
        if resp.status_code == 200:
            data = resp.json()
            print(f"Response: {data.get('response', 'NO RESPONSE')[:500]}")
        else:
            print(f"Error: {resp.text[:500]}")
    except Exception as e:
        print(f"Connection error: {e}")
else:
    print("Unknown endpoint type. Please configure KAGGLE_VLM_URL or OLLAMA_HOST.")
