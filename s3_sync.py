"""S3 persistence for SQLite database and trained model files.

On startup: download DB + model files from S3 to local data/ directory.
After training: upload updated DB + new model file to S3.
"""

import os
from pathlib import Path

import boto3
from botocore.exceptions import ClientError

BUCKET = os.environ.get("S3_BUCKET", "")
DATA_DIR = Path(__file__).parent / "data"
MODELS_DIR = DATA_DIR / "models"
DB_FILENAME = "grade_ninja.db"

_client = None


def _get_client():
    global _client
    if _client is None:
        _client = boto3.client("s3")
    return _client


def _s3_enabled():
    return bool(BUCKET)


def download_all():
    """Download DB and all model files from S3. Called on startup."""
    if not _s3_enabled():
        print("[s3_sync] S3_BUCKET not set, skipping download")
        return

    s3 = _get_client()
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    # Download database
    db_path = DATA_DIR / DB_FILENAME
    try:
        s3.download_file(BUCKET, DB_FILENAME, str(db_path))
        print(f"[s3_sync] Downloaded {DB_FILENAME}")
    except ClientError as e:
        if e.response["Error"]["Code"] == "404":
            print(f"[s3_sync] {DB_FILENAME} not found in S3, starting fresh")
        else:
            print(f"[s3_sync] Error downloading DB: {e}")

    # Download model files
    try:
        resp = s3.list_objects_v2(Bucket=BUCKET, Prefix="models/")
        for obj in resp.get("Contents", []):
            key = obj["Key"]
            local_path = DATA_DIR / key
            local_path.parent.mkdir(parents=True, exist_ok=True)
            s3.download_file(BUCKET, key, str(local_path))
            print(f"[s3_sync] Downloaded {key}")
    except ClientError as e:
        print(f"[s3_sync] Error listing models: {e}")


def upload_db():
    """Upload the SQLite database to S3."""
    if not _s3_enabled():
        return

    s3 = _get_client()
    db_path = DATA_DIR / DB_FILENAME
    if db_path.exists():
        s3.upload_file(str(db_path), BUCKET, DB_FILENAME)
        print(f"[s3_sync] Uploaded {DB_FILENAME}")


def upload_model(session_id: int):
    """Upload a specific model file to S3."""
    if not _s3_enabled():
        return

    s3 = _get_client()
    model_path = MODELS_DIR / f"session_{session_id}.joblib"
    if model_path.exists():
        key = f"models/session_{session_id}.joblib"
        s3.upload_file(str(model_path), BUCKET, key)
        print(f"[s3_sync] Uploaded {key}")
