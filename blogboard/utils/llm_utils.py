from typing import Type
from pydantic import BaseModel
from langchain_groq import ChatGroq
from blogboard.config.settings import app_settings

class GroqClient:
    def __init__(self):
        self.groq_client = ChatGroq(api_key=app_settings.llm.API_KEY, model=app_settings.llm.MODEL_NAME)

    def get_prompt(self, prompt_name: str) -> str:
        with open(f"blogboard/prompts/{prompt_name}.txt", "r", encoding="utf-8") as f:
            return f.read()
    
    def generate_content(self, prompt_name: str, schema: Type[BaseModel], **kwargs: str) -> BaseModel:
        prompt = self.get_prompt(prompt_name)
        structured_llm = self.groq_client.with_structured_output(schema)
        response = structured_llm.invoke(
            input=prompt.format(**kwargs),
        )
        return response