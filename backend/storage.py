import os
import json
import boto3
from botocore.exceptions import ClientError

def get_r2_client():
    """Initializes and returns a boto3 client configured for Cloudflare R2."""
    account_id = os.environ.get("R2_ACCOUNT_ID")
    access_key = os.environ.get("R2_ACCESS_KEY_ID")
    secret_key = os.environ.get("R2_SECRET_ACCESS_KEY")

    if not all([account_id, access_key, secret_key]):
        print("[WARN] R2 credentials not fully set in environment. R2 uploads will fail if not using dry-run.")

    return boto3.client(
        service_name="s3",
        endpoint_url=f"https://{account_id}.r2.cloudflarestorage.com",
        aws_access_key_id=access_key,
        aws_secret_access_key=secret_key,
        region_name="auto" # R2 requires region to be 'auto'
    )

def get_r2_bucket_name():
    return os.environ.get("R2_BUCKET_NAME", "blogboard-bucket")


def get_articles_json_from_r2(domain: str) -> list[dict]:
    """Fetches the current articles.json list for a domain from R2."""
    s3 = get_r2_client()
    bucket = get_r2_bucket_name()
    object_key = f"blogs/{domain}/articles.json"

    try:
        response = s3.get_object(Bucket=bucket, Key=object_key)
        data = response["Body"].read().decode("utf-8")
        return json.loads(data)
    except ClientError as e:
        if e.response['Error']['Code'] == 'NoSuchKey':
            return [] # File doesn't exist yet
        print(f"[ERROR] Failed to fetch {object_key} from R2: {e}")
        return []
    except json.JSONDecodeError:
        print(f"[WARN] {object_key} contained invalid JSON. Starting fresh.")
        return []

def upload_string_to_r2(content: str, object_key: str, content_type: str = "text/plain"):
    """Uploads a raw string to Cloudflare R2."""
    s3 = get_r2_client()
    bucket = get_r2_bucket_name()

    try:
        s3.put_object(
            Bucket=bucket,
            Key=object_key,
            Body=content.encode("utf-8"),
            ContentType=content_type
        )
        print(f"  ✅ Uploaded to R2: {bucket}/{object_key}")
    except ClientError as e:
        print(f"  [ERROR] Failed to upload {object_key} to R2: {e}")
        raise
