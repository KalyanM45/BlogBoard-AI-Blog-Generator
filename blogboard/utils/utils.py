from blogboard.clients.manager import client_manager

def get_prompt(name: str) -> str:
    prompt_obj = client_manager.langfuse.langfuse_client.get_prompt(name=name)
    return prompt_obj.compile()