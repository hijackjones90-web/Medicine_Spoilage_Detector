"""SageMaker inference entry point for Qwen2.5-VL-7B.

This script runs inside the SageMaker container and handles:
- Model loading with 4-bit quantization (bitsandbytes)
- Request/response handling via FastAPI
- Base64 image decoding and processing
- Qwen2.5-VL chat template with image support

Docker image must include:
- transformers>=4.45.0
- torch>=2.0.0
- bitsandbytes>=0.43.1
- accelerate>=0.30.0
- fastapi>=0.100.0
- uvicorn>=0.23.0
- pillow>=10.0.0
"""

import os
import json
import base64
import io
import logging
from typing import List

import torch
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from PIL import Image
from transformers import (
    AutoProcessor,
    Qwen2_5_VLForConditionalGeneration,
    BitsAndBytesConfig,
)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global model/processor (loaded once on container start)
MODEL = None
PROCESSOR = None
MODEL_ID = os.environ.get("MODEL_ID", "Qwen/Qwen2.5-VL-7B-Instruct")


class InferenceRequest(BaseModel):
    """Request format matching Modal's analyze endpoint."""
    images: List[str] = Field(..., description="Base64-encoded JPEG images")
    prompt: str = Field(..., description="Text prompt for the VLM")


class InferenceResponse(BaseModel):
    """Response format matching Modal's analyze endpoint."""
    response: str


def load_model():
    """Load the quantized model and processor."""
    global MODEL, PROCESSOR
    
    if MODEL is not None:
        return
    
    logger.info(f"Loading model: {MODEL_ID}")
    
    # 4-bit quantization config
    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
    )
    
    PROCESSOR = AutoProcessor.from_pretrained(
        MODEL_ID,
        trust_remote_code=True,
    )
    
    MODEL = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        MODEL_ID,
        quantization_config=quantization_config,
        device_map="auto",
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
    )
    
    logger.info("Model loaded successfully")


def decode_base64_image(b64_string: str) -> Image.Image:
    """Decode base64 string to PIL Image."""
    try:
        image_data = base64.b64decode(b64_string)
        image = Image.open(io.BytesIO(image_data))
        if image.mode != "RGB":
            image = image.convert("RGB")
        return image
    except Exception as e:
        logger.error(f"Failed to decode base64 image: {e}")
        raise HTTPException(status_code=400, detail="Invalid base64 image")


# Initialize FastAPI app
app = FastAPI(title="Qwen2.5-VL SageMaker Endpoint")


@app.on_event("startup")
async def startup_event():
    """Load model on container startup."""
    load_model()


@app.get("/ping")
async def ping():
    """Health check endpoint."""
    return {"status": "healthy", "model_loaded": MODEL is not None}


@app.post("/invocations", response_model=InferenceResponse)
async def invocations(request: InferenceRequest):
    """
    Main inference endpoint.
    
    Accepts:
    {
        "images": ["base64_jpeg1", "base64_jpeg2", ...],
        "prompt": "Your prompt here"
    }
    
    Returns:
    {
        "response": "Model response text"
    }
    """
    if MODEL is None or PROCESSOR is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    try:
        # Decode images
        images = [decode_base64_image(b64) for b64 in request.images]
        
        # Build message content
        content = [{"type": "text", "text": request.prompt}]
        for img in images:
            # Save image to bytes and re-encode for processor
            buf = io.BytesIO()
            img.save(buf, format="JPEG", quality=95)
            img_b64 = base64.b64encode(buf.getvalue()).decode()
            content.append({
                "type": "image",
                "image": img,
            })
        
        messages = [{"role": "user", "content": content}]
        
        # Apply chat template
        text = PROCESSOR.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        
        # Process inputs
        inputs = PROCESSOR(
            text=[text],
            images=images if images else None,
            return_tensors="pt",
            padding=True,
        ).to(MODEL.device)
        
        # Generate
        with torch.no_grad():
            output_ids = MODEL.generate(
                **inputs,
                max_new_tokens=4096,
                temperature=0.1,
                do_sample=True,
                top_p=0.95,
            )
        
        # Decode only the generated part
        generated_ids = output_ids[:, inputs["input_ids"].shape[1]:]
        response = PROCESSOR.batch_decode(
            generated_ids,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )[0]
        
        return InferenceResponse(response=response)
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Inference error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8080)