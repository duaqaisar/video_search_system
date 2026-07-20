"""
S3-compatible storage (MinIO locally, real S3 in production).
Same boto3 client works for both — just swap the endpoint_url and
credentials via environment variables when deploying.
"""
import os
import boto3
from botocore.exceptions import ClientError
from pathlib import Path

S3_ENDPOINT_URL = os.environ.get("S3_ENDPOINT_URL", "http://localhost:9000")
S3_ACCESS_KEY = os.environ.get("S3_ACCESS_KEY", "videoapp")
S3_SECRET_KEY = os.environ.get("S3_SECRET_KEY", "videoapp_dev_password")
S3_BUCKET = os.environ.get("S3_BUCKET", "videos")

_client = None


def _get_client():
    global _client
    if _client is None:
        _client = boto3.client(
            "s3",
            endpoint_url=S3_ENDPOINT_URL,
            aws_access_key_id=S3_ACCESS_KEY,
            aws_secret_access_key=S3_SECRET_KEY,
        )
    return _client


def ensure_bucket():
    client = _get_client()
    try:
        client.head_bucket(Bucket=S3_BUCKET)
    except ClientError:
        client.create_bucket(Bucket=S3_BUCKET)
        print(f"Created bucket '{S3_BUCKET}'")


def upload_file(local_path: Path, storage_key: str) -> str:
    """Uploads a local file to storage, returns the storage key."""
    client = _get_client()
    ensure_bucket()
    client.upload_file(str(local_path), S3_BUCKET, storage_key)
    return storage_key


def download_file(storage_key: str, local_path: Path) -> Path:
    """Downloads a file from storage to a local path."""
    client = _get_client()
    local_path.parent.mkdir(parents=True, exist_ok=True)
    client.download_file(S3_BUCKET, storage_key, str(local_path))
    return local_path


def generate_presigned_url(storage_key: str, expires_in: int = 3600) -> str:
    """Generates a temporary signed URL for direct browser access to a file."""
    client = _get_client()
    return client.generate_presigned_url(
        "get_object",
        Params={"Bucket": S3_BUCKET, "Key": storage_key},
        ExpiresIn=expires_in,
    )
