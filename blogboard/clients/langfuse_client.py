from langfuse import Langfuse
from langfuse.langchain import CallbackHandler
from blogboard.config.settings import app_settings

class LangfuseClient:
    """A unified wrapper for the Langfuse SDK, handling initialization, prompt-fetching, callbacks, and graceful flushing."""
    
    def initialize_langfuse(self):
        self.langfuse_client = Langfuse(
            public_key=app_settings.langfuse.PUBLIC_KEY,
            secret_key=app_settings.langfuse.SECRET_KEY,
            base_url=app_settings.langfuse.BASE_URL
        )
        self.langfuse_callback_handler = CallbackHandler()


    def flush_langfuse(self):
        """Ensure all local traces and observations are sent to the remote host."""
        if self.langfuse_client:
            self.langfuse_client.flush()
        
        if hasattr(self.langfuse_callback_handler, 'flush'):
            self.langfuse_callback_handler.flush()
