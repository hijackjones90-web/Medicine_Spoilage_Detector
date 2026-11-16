"""Deploy Qwen2.5-VL-7B to SageMaker.

This script creates a SageMaker model, endpoint configuration, and endpoint
for running Qwen2.5-VL-7B with 4-bit quantization.

Prerequisites:
- AWS CLI configured with appropriate permissions
- SageMaker execution role ARN
- ECR repository for custom inference container
- S3 bucket for model artifacts (optional, can use HF Hub directly)

Usage:
    python deploy_sagemaker.py \
        --role-arn arn:aws:iam::123456789012:role/SageMakerExecutionRole \
        --endpoint-name qwen-vl-endpoint \
        --instance-type ml.g4dn.xlarge
"""

import argparse
import boto3
import json
import logging
import os
import subprocess
import sys
import time

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def get_account_id():
    """Get current AWS account ID."""
    sts = boto3.client("sts")
    return sts.get_caller_identity()["Account"]


def get_region():
    """Get current AWS region."""
    session = boto3.session.Session()
    return session.region_name or "us-east-1"


def build_and_push_docker_image(ecr_repo: str, region: str, account_id: str) -> str:
    """Build and push the SageMaker inference Docker image to ECR."""
    image_uri = f"{account_id}.dkr.ecr.{region}.amazonaws.com/{ecr_repo}:latest"
    
    # Create ECR repo if it doesn't exist
    ecr = boto3.client("ecr", region_name=region)
    try:
        ecr.describe_repositories(repositoryNames=[ecr_repo])
        logger.info(f"ECR repository {ecr_repo} already exists")
    except ecr.exceptions.RepositoryNotFoundException:
        logger.info(f"Creating ECR repository: {ecr_repo}")
        ecr.create_repository(repositoryName=ecr_repo)
    
    # Get ECR login token
    token = ecr.get_authorization_token(registryIds=[account_id])
    auth = token["authorizationData"][0]
    import base64
    username, password = base64.b64decode(auth["authorizationToken"]).decode().split(":")
    registry = auth["proxyEndpoint"]
    
    # Login to ECR
    subprocess.run(
        ["docker", "login", "-u", username, "-p", password, registry],
        check=True,
    )
    
    # Build Docker image
    dockerfile = """
FROM python:3.11-slim

# Install system dependencies
RUN apt-get update && apt-get install -y \\
    git \\
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
RUN pip install --no-cache-dir \\
    transformers==4.45.2 \\
    torch==2.3.0 \\
    accelerate==0.30.1 \\
    bitsandbytes==0.43.1 \\
    fastapi==0.111.0 \\
    uvicorn==0.30.1 \\
    pillow==10.3.0 \\
    sentencepiece==0.1.99

# Copy inference script
COPY sagemaker_inference.py /opt/ml/code/sagemaker_inference.py

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

# SageMaker expects the handler in /opt/ml/code
WORKDIR /opt/ml/code

EXPOSE 8080

CMD ["python", "sagemaker_inference.py"]
"""
    
    # Write Dockerfile
    with open("Dockerfile.sagemaker", "w") as f:
        f.write(dockerfile)
    
    logger.info("Building Docker image...")
    subprocess.run(
        ["docker", "build", "-f", "Dockerfile.sagemaker", "-t", image_uri, "."],
        check=True,
    )
    
    logger.info("Pushing Docker image to ECR...")
    subprocess.run(["docker", "push", image_uri], check=True)
    
    return image_uri


def create_model(sagemaker_client, model_name: str, image_uri: str, role_arn: str):
    """Create SageMaker model."""
    logger.info(f"Creating SageMaker model: {model_name}")
    
    try:
        sagemaker_client.create_model(
            ModelName=model_name,
            ExecutionRoleArn=role_arn,
            PrimaryContainer={
                "Image": image_uri,
                "Mode": "SingleModel",
                "Environment": {
                    "MODEL_ID": "Qwen/Qwen2.5-VL-7B-Instruct",
                    "SAGEMAKER_CONTAINER_LOG_LEVEL": "INFO",
                    "SAGEMAKER_REGION": get_region(),
                },
            },
        )
        logger.info(f"Model {model_name} created successfully")
    except sagemaker_client.exceptions.ClientError as e:
        if "AlreadyExists" in str(e):
            logger.info(f"Model {model_name} already exists")
        else:
            raise


def create_endpoint_config(sagemaker_client, config_name: str, model_name: str, instance_type: str):
    """Create endpoint configuration."""
    logger.info(f"Creating endpoint config: {config_name}")
    
    try:
        sagemaker_client.create_endpoint_config(
            EndpointConfigName=config_name,
            ProductionVariants=[
                {
                    "VariantName": "primary",
                    "ModelName": model_name,
                    "InitialInstanceCount": 1,
                    "InstanceType": instance_type,
                    "InitialVariantWeight": 1.0,
                }
            ],
        )
        logger.info(f"Endpoint config {config_name} created successfully")
    except sagemaker_client.exceptions.ClientError as e:
        if "AlreadyExists" in str(e):
            logger.info(f"Endpoint config {config_name} already exists")
        else:
            raise


def create_endpoint(sagemaker_client, endpoint_name: str, config_name: str):
    """Create or update SageMaker endpoint."""
    logger.info(f"Creating endpoint: {endpoint_name}")
    
    try:
        sagemaker_client.create_endpoint(
            EndpointName=endpoint_name,
            EndpointConfigName=config_name,
        )
    except sagemaker_client.exceptions.ClientError as e:
        if "AlreadyExists" in str(e):
            logger.info(f"Endpoint {endpoint_name} exists, updating...")
            sagemaker_client.update_endpoint(
                EndpointName=endpoint_name,
                EndpointConfigName=config_name,
            )
        else:
            raise


def wait_for_endpoint(sagemaker_client, endpoint_name: str, timeout: int = 1200):
    """Wait for endpoint to become InService."""
    logger.info(f"Waiting for endpoint {endpoint_name} to be InService...")
    start = time.time()
    
    while time.time() - start < timeout:
        response = sagemaker_client.describe_endpoint(EndpointName=endpoint_name)
        status = response["EndpointStatus"]
        
        if status == "InService":
            logger.info(f"Endpoint {endpoint_name} is InService!")
            return True
        elif status == "Failed":
            failure_reason = response.get("FailureReason", "Unknown")
            logger.error(f"Endpoint creation failed: {failure_reason}")
            return False
        else:
            logger.info(f"Endpoint status: {status} (waiting...)")
            time.sleep(30)
    
    logger.error(f"Timeout waiting for endpoint {endpoint_name}")
    return False


def get_endpoint_url(endpoint_name: str, region: str) -> str:
    """Construct the SageMaker endpoint invoke URL."""
    return f"https://runtime.sagemaker.{region}.amazonaws.com/endpoints/{endpoint_name}/invocations"


def main():
    parser = argparse.ArgumentParser(description="Deploy Qwen2.5-VL to SageMaker")
    parser.add_argument("--role-arn", required=True, help="SageMaker execution role ARN")
    parser.add_argument("--endpoint-name", default="qwen-vl-endpoint", help="Endpoint name")
    parser.add_argument("--model-name", default="qwen-vl-model", help="Model name")
    parser.add_argument("--config-name", default="qwen-vl-config", help="Endpoint config name")
    parser.add_argument("--ecr-repo", default="qwen-vl-inference", help="ECR repository name")
    parser.add_argument("--instance-type", default="ml.g4dn.xlarge", help="Instance type")
    parser.add_argument("--region", default=None, help="AWS region (default: from AWS config)")
    parser.add_argument("--skip-docker", action="store_true", help="Skip Docker build/push (use existing image)")
    parser.add_argument("--image-uri", default=None, help="Pre-built image URI (use with --skip-docker)")
    
    args = parser.parse_args()
    
    region = args.region or get_region()
    account_id = get_account_id()
    
    logger.info(f"Deploying to region: {region}, account: {account_id}")
    logger.info(f"Endpoint: {args.endpoint_name}")
    logger.info(f"Instance type: {args.instance_type}")
    
    sagemaker = boto3.client("sagemaker", region_name=region)
    
    # Build and push Docker image
    if not args.skip_docker:
        image_uri = build_and_push_docker_image(args.ecr_repo, region, account_id)
    else:
        image_uri = args.image_uri or f"{account_id}.dkr.ecr.{region}.amazonaws.com/{args.ecr_repo}:latest"
    
    logger.info(f"Using image: {image_uri}")
    
    # Create model
    create_model(sagemaker, args.model_name, image_uri, args.role_arn)
    
    # Create endpoint config
    create_endpoint_config(sagemaker, args.config_name, args.model_name, args.instance_type)
    
    # Create/update endpoint
    create_endpoint(sagemaker, args.endpoint_name, args.config_name)
    
    # Wait for endpoint to be ready
    if wait_for_endpoint(sagemaker, args.endpoint_name):
        endpoint_url = get_endpoint_url(args.endpoint_name, region)
        logger.info(f"Endpoint ready at: {endpoint_url}")
        logger.info(f"Set SAGEMAKER_ENDPOINT_URL={endpoint_url} in your environment")
        
        # Save endpoint info
        with open("sagemaker_endpoint.json", "w") as f:
            json.dump({
                "endpoint_name": args.endpoint_name,
                "endpoint_url": endpoint_url,
                "region": region,
                "model_name": args.model_name,
                "config_name": args.config_name,
            }, f, indent=2)
    else:
        logger.error("Deployment failed")
        sys.exit(1)


if __name__ == "__main__":
    main()