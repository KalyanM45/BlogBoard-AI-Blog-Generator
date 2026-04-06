import json
from blogboard.config.settings import app_settings
from blogboard.client.client_manager import manager
from botocore.exceptions import ClientError

class DBUtils:
    def __init__(self):
        self.s3 = manager.get_r2_client()

    def get_data(self, path: str):
        bucket_name = app_settings.r2.BUCKET_NAME.strip(' ="\'')
        try:
            response = self.s3.get_object(Bucket=bucket_name, Key=path)

            return response["Body"].read().decode("utf-8")
        except ClientError as e:
            if e.response['Error']['Code'] == 'NoSuchKey':
                return None
            print(f"[ERROR] R2 error in get_data: {e}")
            return None
        except Exception as e:
            print(f"[ERROR] Unexpected error in get_data: {e}")
            return None

    def upload_data(self, path: str, data: str, content_type: str = "text/plain"):
        bucket_name = app_settings.r2.BUCKET_NAME.strip(' ="\'')
        try:
            self.s3.put_object(
                Bucket=bucket_name, 
                Key=path, 
                Body=data.encode("utf-8"),
                ContentType=content_type
            )
            return True

        except Exception as e:
            print(f"[ERROR] Unexpected error in upload_data: {e}")
            return False

    def get_json(self, path: str):
        data = self.get_data(path)
        if data:
            try:
                return json.loads(data)
            except json.JSONDecodeError:
                print(f"[WARN] Failed to decode JSON from {path}")
                return None
        return None

    def get_recent_blog_topics(self, domain: str, limit: int = 3) -> str:
        try:
            articles = self.get_json(f"blogs/{domain}/articles.json")
            if not articles:
                return "No recent topics found."
            
            sorted_articles = sorted(articles, key=lambda x: x.get("date", ""), reverse=True)
            recent = sorted_articles[:limit]
            topics = [a.get("topic") for a in recent if a.get("topic")]
            return "\n".join(topics) if topics else "No recent topics found."

        except Exception as e:
            print(f"[ERROR] Unexpected error in get_recent_blog_topics: {e}")
            return "Error retrieving history."

    def get_recent_history(self, domain: str, limit: int = 3) -> list[dict]:
        """Fetches the N most recent articles for a specific domain to give context to the LLM."""
        articles = self.get_json(f"blogs/{domain}/articles.json")
        if not articles:
            return []
            
        sorted_articles = sorted(articles, key=lambda x: x.get("date", ""), reverse=True)
        recent = sorted_articles[:limit]
        
        # Prune heavy data to save on prompt tokens
        history_summary = []
        for a in recent:
            history_summary.append({
                "title": a.get("title"),
                "topic": a.get("topic"),
                "subtopics": a.get("subtopics", "")
            })
        return history_summary

    def get_all_domains_last_updated(self) -> dict[str, str]:
        from blogboard.config.config import DOMAIN_MAP
        
        latest_dates = {}
        for domain in DOMAIN_MAP.values():
            articles = self.get_json(f"blogs/{domain}/articles.json")
            if not articles:
                latest_dates[domain] = "Never"
                continue
                
            sorted_articles = sorted(articles, key=lambda x: x.get("date", ""), reverse=True)
            latest_dates[domain] = sorted_articles[0].get("date", "Unknown")
            
        return latest_dates
