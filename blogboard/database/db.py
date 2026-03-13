import json
import boto3
from botocore.exceptions import ClientError
from blogboard.config.settings import app_settings

def get_r2_client():
    """Initializes and returns a boto3 client for Cloudflare R2."""
    return boto3.client(
        service_name="s3",
        endpoint_url=f"https://{app_settings.r2.ACCOUNT_ID}.r2.cloudflarestorage.com",
        aws_access_key_id=app_settings.r2.ACCESS_KEY_ID,
        aws_secret_access_key=app_settings.r2.SECRET_ACCESS_KEY,
        region_name="auto"
    )

def get_domain_slug(domain_name: str) -> str:
    """Maps a full domain label back to its short slug (e.g., 'Machine Learning' -> 'ml')."""
    for field_name, tag_info in app_settings.tags.model_dump().items():
        if tag_info.get("label") == domain_name:
            return field_name
    # Fallback to slugified version if no match found
    return domain_name.lower().replace(" ", "_")

def get_recent_blog_topics(domain: str, limit: int = 3) -> str:
    """
    Retrieves the last N blog topics for a specific domain from R2.
    """
    domain_slug = get_domain_slug(domain)
    
    s3 = get_r2_client()
    bucket = app_settings.r2.BUCKET_NAME
    # Path: blogs/{domain}/articles.json
    object_key = f"blogs/{domain_slug}/articles.json"

    try:
        response = s3.get_object(Bucket=bucket, Key=object_key)
        data = response["Body"].read().decode("utf-8")
        articles = json.loads(data)
        
        # Sort by date descending
        sorted_articles = sorted(articles, key=lambda x: x.get("date", ""), reverse=True)
        recent = sorted_articles[:limit]
        
        # Extract topics (using 'topic' field as per user's structure)
        topics = [a.get("topic") for a in recent if a.get("topic")]
        return "\n".join(topics) if topics else "No recent topics found."

    except ClientError as e:
        if e.response['Error']['Code'] == 'NoSuchKey':
            return "No previous articles found for this domain."
        print(f"[ERROR] R2 fetch failed for {object_key}: {e}")
        return "Error retrieving history."
    except Exception as e:
        print(f"[ERROR] Unexpected error in get_recent_blog_topics: {e}")
        return "Error retrieving history."
