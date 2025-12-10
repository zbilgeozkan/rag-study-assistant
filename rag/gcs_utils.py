import os
from google.cloud import storage

# Get from env or use default bucket
GCS_BUCKET_NAME = os.getenv("GCS_BUCKET_NAME", "rag-documents-bucket-icu")

def get_storage_client():
    return storage.Client()

def upload_file_to_gcs(local_path: str, gcs_path: str, bucket_name: str | None = None):
    """
    local_path -> GCS (bucket/gcs_path)
    """
    bucket_name = bucket_name or GCS_BUCKET_NAME
    client = get_storage_client()
    bucket = client.bucket(bucket_name)
    blob = bucket.blob(gcs_path)

    blob.upload_from_filename(local_path)
    print(f"[GCS] Uploaded {local_path} -> gs://{bucket_name}/{gcs_path}")

def download_file_from_gcs(gcs_path: str, local_path: str, bucket_name: str | None = None):
    """
    GCS -> local_path
    """
    bucket_name = bucket_name or GCS_BUCKET_NAME
    client = get_storage_client()
    bucket = client.bucket(bucket_name)
    blob = bucket.blob(gcs_path)

    os.makedirs(os.path.dirname(local_path), exist_ok=True)
    blob.download_to_filename(local_path)
    print(f"[GCS] Downloaded gs://{bucket_name}/{gcs_path} -> {local_path}")

def file_exists_in_gcs(gcs_path: str, bucket_name: str | None = None) -> bool:
    bucket_name = bucket_name or GCS_BUCKET_NAME
    client = get_storage_client()
    bucket = client.bucket(bucket_name)
    blob = bucket.blob(gcs_path)
    return blob.exists()
