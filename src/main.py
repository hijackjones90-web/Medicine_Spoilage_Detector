"""Main entry point for SageMaker deployment of Qwen2.5-VL-7B.

This file is used by SageMaker to host the inference endpoint.
When SageMaker starts this container, it executes this file and runs the FastAPI app.

Note: SageMaker loads the model once on container startup, then
receives requests via the /invocations endpoint.
"""

from sagemaker_inference import app

# SageMaker expects the Flask/FastAPI app to be named 'app'
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8080)