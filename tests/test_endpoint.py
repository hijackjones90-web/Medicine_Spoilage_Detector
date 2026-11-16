"""Quick test of the Modal endpoint."""
import requests
import json
import base64
import sys
import os
from PIL import Image
import io

url = os.environ.get("MODAL_ENDPOINT_URL", "")
if not url:
    print("ERROR: MODAL_ENDPOINT_URL is not set. Deploy backend.py and export the URL.")
    sys.exit(1)

# Create a small test image
img = Image.new("RGB", (100, 100), color=(255, 200, 150))
buf = io.BytesIO()
img.save(buf, format="JPEG", quality=95)
b64 = base64.b64encode(buf.getvalue()).decode()

# Modal endpoint: prompt in query params, images list as raw JSON body
prompt = 'Return ONLY valid JSON: {"status": "ok", "model": "working"}'

print(f"Sending test request to: {url}")
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
        print(f"Response: {data.get('response', 'NO RESPONSE KEY')[:500]}")
    else:
        print(f"Error body: {resp.text[:500]}")
except Exception as e:
    print(f"Connection error: {e}")