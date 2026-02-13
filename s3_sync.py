"""S3 persistence for SQLite database and trained model files.

On startup: download DB + model files from S3 to local data/ directory.
After training: upload updated DB + new model file to S3.
"""

import os
from pathlib import Path

BUCKET = os.environ.get("S3_BUCKET", "")
DATA_DIR = Path(__file__).parent / "data"
MODELS_DIR = DATA_DIR / "models"
DB_FILENAME = "grade_ninja.db"

_client = None


def _get_client():
    global _client
    if _client is None:
        import boto3
        _client = boto3.client("s3")
    return _client


def _s3_enabled():
    return bool(BUCKET)


def download_all():
    """Download DB and all model files from S3. Called on startup.
    Never raises — logs errors and continues so the app always starts.
    """
    if not _s3_enabled():
        print("[s3_sync] S3_BUCKET not set, skipping download")
        return

    try:
        s3 = _get_client()
        DATA_DIR.mkdir(parents=True, exist_ok=True)
        MODELS_DIR.mkdir(parents=True, exist_ok=True)

        # Download database
        db_path = DATA_DIR / DB_FILENAME
        try:
            s3.head_object(Bucket=BUCKET, Key=DB_FILENAME)
            s3.download_file(BUCKET, DB_FILENAME, str(db_path))
            print(f"[s3_sync] Downloaded {DB_FILENAME}")
        except Exception:
            print(f"[s3_sync] {DB_FILENAME} not found in S3, starting fresh")

        # Download model files
        try:
            resp = s3.list_objects_v2(Bucket=BUCKET, Prefix="models/")
            for obj in resp.get("Contents", []):
                key = obj["Key"]
                if key.endswith("/"):
                    continue
                local_path = DATA_DIR / key
                local_path.parent.mkdir(parents=True, exist_ok=True)
                s3.download_file(BUCKET, key, str(local_path))
                print(f"[s3_sync] Downloaded {key}")
        except Exception as e:
            print(f"[s3_sync] Error listing models: {e}")

    except Exception as e:
        print(f"[s3_sync] S3 download failed, starting fresh: {e}")


def upload_db():
    """Upload the SQLite database to S3."""
    if not _s3_enabled():
        return

    try:
        s3 = _get_client()
        db_path = DATA_DIR / DB_FILENAME
        if db_path.exists():
            s3.upload_file(str(db_path), BUCKET, DB_FILENAME)
            print(f"[s3_sync] Uploaded {DB_FILENAME}")
    except Exception as e:
        print(f"[s3_sync] Error uploading DB: {e}")


def upload_model(session_id: int):
    """Upload a specific model file to S3."""
    if not _s3_enabled():
        return

    try:
        s3 = _get_client()
        model_path = MODELS_DIR / f"session_{session_id}.joblib"
        if model_path.exists():
            key = f"models/session_{session_id}.joblib"
            s3.upload_file(str(model_path), BUCKET, key)
            print(f"[s3_sync] Uploaded {key}")
    except Exception as e:
        print(f"[s3_sync] Error uploading model: {e}")
