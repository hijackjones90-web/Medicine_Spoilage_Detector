"""Test the full pipeline with a medicine image via SageMaker."""
import requests
import json
import base64
from PIL import Image
import io
import os

url = os.environ.get("SAGEMAKER_ENDPOINT_URL", "")
if not url:
    print("ERROR: SAGEMAKER_ENDPOINT_URL is not set. Deploy to SageMaker first.")
    import sys
    sys.exit(1)

# Create a test image (orange liquid like Jectocos Lipo)
img = Image.new("RGB", (200, 200), color=(255, 140, 50))
buf = io.BytesIO()
img.save(buf, format="JPEG", quality=95)
b64 = base64.b64encode(buf.getvalue()).decode()

def test_prompt(prompt, label):
    print(f"=== {label} ===")
    try:
        resp = requests.post(url, json={"images": [b64], "prompt": prompt}, timeout=300)
        print(f"Status: {resp.status_code}")
        if resp.status_code == 200:
            data = resp.json()
            response = data.get("response", "")
            print(f"Response length: {len(response)} chars")
            print(f"Response preview: {response[:300]}")
        else:
            print(f"Error: {resp.text[:200]}")
    except Exception as e:
        print(f"Error: {e}")

ocr_prompt = """Read ALL visible text from this medicine packaging. Check front, sides, back, stickers, caps, and foil.

For each text block, return:
- text: exact text as shown
- location: front|side_panel|back|sticker|cap|foil|other
- source_image: photo number (1, 2, 3...)

Also identify:
- packaging_type: bottle|box|strip|blister|tube|sachet|other
- surfaces_visible: list of visible surfaces

Return ONLY valid JSON:
{"text_blocks": [{"text": "exact text", "location": "sticker", "source_image": 1}], "packaging_type": "bottle", "surfaces_visible": ["front", "sticker"]}
"""

info_prompt = """Using the OCR text below, extract medicine information.

Return ONLY valid JSON:
{"name": "product name", "manufacturer": "company", "mfg_date": "date string or null", "exp_date": "date string or null", "batch_no": "batch number or null", "dosage_form": "syrup|tablet|capsule|other", "ingredients": ["list"], "storage_conditions": "text or null"}

If a field is not found, use null. Do not guess.

OCR text:
[photo 1, sticker] B.No: NJL25004 MFG: SEP 2025 EXP: FEB 2027
[photo 1, front] Jectocos Lipo - Iron, Folic Acid and Vitamin B12
"""

spoilage_prompt = """Look at this medicine image. Assess visual spoilage.

Return ONLY valid JSON:
{"discoloration": true/false, "cloudiness": true/false, "sediment": true/false, "seal_intact": true/false, "label_damage": true/false, "spoilage_level": 0-100, "indicators": ["list"]}

Note: Suspensions (like iron syrup) are NATURALLY opaque. Do not flag that as cloudiness.
"""

test_prompt(ocr_prompt, "Test 1: OCR Prompt")
test_prompt(info_prompt, "Test 2: Info Prompt")
test_prompt(spoilage_prompt, "Test 3: Spoilage Prompt")