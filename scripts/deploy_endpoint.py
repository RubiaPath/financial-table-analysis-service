#!/usr/bin/env python3
import os
import sys
import time
import boto3
from botocore.exceptions import ClientError

AWS_REGION = os.getenv("AWS_REGION", "ap-northeast-1")
ROLE_ARN = os.environ.get("ROLE_ARN")
IMAGE_URI = os.environ.get("IMAGE_URI")

if not ROLE_ARN or not IMAGE_URI:
    print("ERROR: ROLE_ARN and IMAGE_URI required")
    sys.exit(1)

MODEL_DATA_URL = os.getenv("MODEL_DATA_URL", "s3://table-analysis-storage-models/model.tar.gz")
INSTANCE_TYPE = os.getenv("INSTANCE_TYPE", "ml.g4dn.xlarge")
MODEL_NAME = os.getenv("MODEL_NAME", "financial-table-analysis-model")
EPC_NAME = os.getenv("EPC_NAME", "financial-table-analysis-epc")
ENDPOINT_NAME = os.getenv("ENDPOINT_NAME", "financial-table-analysis")

sm = boto3.client("sagemaker", region_name=AWS_REGION)

print(f"Region: {AWS_REGION}")
print(f"Image: {IMAGE_URI}")
print(f"Endpoint: {ENDPOINT_NAME}")
print()

def create_model():
    try:
        sm.create_model(
            ModelName=MODEL_NAME,
            PrimaryContainer={
                "Image": IMAGE_URI,
                "ModelDataUrl": MODEL_DATA_URL,
                "Environment": {
                    "SM_MODEL_DIR": "/opt/ml/model",
                    "OLLAMA_MODELS": "/opt/ml/model/ollama/models",
                    "SAM3_CHECKPOINT_DIR": "/opt/ml/model/sam3/checkpoints",
                },
            },
            ExecutionRoleArn=ROLE_ARN,
        )
        print("Model created")
    except ClientError as e:
        if e.response["Error"]["Code"] in ("ValidationException", "ResourceInUse"):
            print("Model exists")
        else:
            raise

def create_endpoint_config():
    try:
        sm.create_endpoint_config(
            EndpointConfigName=EPC_NAME,
            ProductionVariants=[{
                "VariantName": "Primary",
                "ModelName": MODEL_NAME,
                "InstanceType": INSTANCE_TYPE,
                "InitialInstanceCount": 1,
            }],
        )
        print("Endpoint config created")
    except ClientError as e:
        if e.response["Error"]["Code"] in ("ValidationException", "ResourceInUse"):
            print("Endpoint config exists")
        else:
            raise

def create_or_update_endpoint():
    try:
        sm.describe_endpoint(EndpointName=ENDPOINT_NAME)
        print("Updating endpoint...")
        sm.update_endpoint(EndpointName=ENDPOINT_NAME, EndpointConfigName=EPC_NAME)
    except ClientError as e:
        if e.response["Error"]["Code"] == "ValidationException":
            print("Creating endpoint...")
            sm.create_endpoint(EndpointName=ENDPOINT_NAME, EndpointConfigName=EPC_NAME)
        else:
            raise

def wait_ready():
    print("Waiting for endpoint...")
    for i in range(120):
        resp = sm.describe_endpoint(EndpointName=ENDPOINT_NAME)
        status = resp["EndpointStatus"]
        print(f"  Status: {status}")
        
        if status == "InService":
            print("READY")
            return
        elif status in ("Failed", "UpdateFailed"):
            print(f"FAILED: {resp.get('FailureReason', 'unknown')}")
            sys.exit(1)
        
        time.sleep(30)
    
    print("TIMEOUT")
    sys.exit(1)

if __name__ == "__main__":
    create_model()
    create_endpoint_config()
    create_or_update_endpoint()
    wait_ready()
