from blogboard.clients.langfuse_client import LangfuseClient

class ClientManager:
    """
    A singleton-like registry for various external observability and telemetry clients.
    Initializes them lazily only when accessed.
    """
    def __init__(self):
        self.langfuse = LangfuseClient()

    def initialize_langfuse(self):
        return self.langfuse.initialize_langfuse()
        
    def flush_langfuse(self):
        """Synchronously flush events from all initialized clients before project exit."""
        return self.langfuse.flush_langfuse()

# Expose a global instance
client_manager = ClientManager()
