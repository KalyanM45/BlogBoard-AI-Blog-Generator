import os
import logging
from blogboard.clients import client_manager

logger = logging.getLogger(__name__)

class PromptManager:
    def get_prompt(self, prompt_name: str, **kwargs) -> str:
        """
        Fetch and format a prompt from the designated metrics/observability platform (Langfuse).
        Local fallbacks are disabled as per project requirements.
        """
        try:
            prompt_obj = client_manager.langfuse.langfuse_client.get_prompt(name=prompt_name)
            return prompt_obj.compile(**kwargs)
        except Exception as e:
            logger.error(f"❌ Critical failure: Unable to fetch or format prompt '{prompt_name}': {e}")
            raise RuntimeError(f"Prompt fetch failed for '{prompt_name}': {e}")

prompt_manager = PromptManager()