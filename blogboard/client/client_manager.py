import boto3
from blogboard.config.settings import app_settings

class ClientManager:
    def __init__(self):
        self.r2_client = None

    def get_r2_client(self):
        """Initializes and returns a boto3 client for Cloudflare R2."""
        if self.r2_client is None:
            self.r2_client = boto3.client(
                service_name="s3",
                endpoint_url=f"https://{app_settings.r2.ACCOUNT_ID}.r2.cloudflarestorage.com",
                aws_access_key_id=app_settings.r2.ACCESS_KEY_ID,
                aws_secret_access_key=app_settings.r2.SECRET_ACCESS_KEY,
                region_name="auto"
            )
        return self.r2_client

manager = ClientManager()