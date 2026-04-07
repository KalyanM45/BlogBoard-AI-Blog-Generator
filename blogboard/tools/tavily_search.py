from typing import Type
from pydantic import BaseModel, Field
from langchain_core.tools import BaseTool

from blogboard.config.settings import app_settings
import requests
from typing import Type
from pydantic import BaseModel, Field
from langchain_core.tools import BaseTool


class TavilySearchInput(BaseModel):
    """Input schema for the TavilySearchTool."""
    query: str = Field(description="The search query to look up on the web.")
    days: int = Field(default=7, description="Number of days to look back for recent news.")


class TavilySearchTool(BaseTool):
    """
    Tavily Search Tool.
    
    This tool utilizes the Tavily API to fetch the most recent news articles
    based on a specific query. It is designed to be used by Langchain agents.
    """
    name: str = "tavily_search"
    description: str = "Search the web for news articles using the Tavily API. Useful for getting up-to-date technical or general news."
    args_schema: Type[BaseModel] = TavilySearchInput

    # Properly configuring the Pydantic behaviour for LangChain's BaseTool
    model_config = {
        "extra": "ignore" 
    }

    def _run(self, query: str, days: int = 7) -> str:
        """
        Executes the search query against the Tavily API.
        
        Args:
            query (str): The search phrase.
            days (int): Number of days to look back.
            
        Returns:
            str: A formatted string containing the top search results.
        """
        api_key = app_settings.content.TAVILY_API_KEY
        if not api_key:
             return "Error: TAVILY_API_KEY is not configured in the application settings."

        try:
            response = requests.post(
                "https://api.tavily.com/search",
                json={
                    "api_key": api_key,
                    "query": query,
                    "topic": "news",
                    "days": days,
                    "max_results": 3,
                    "include_raw_content": False,
                    "include_images": False
                },
                timeout=10
            )
            response.raise_for_status()
            data = response.json()
            
            results = []
            for result in data.get("results", []):
                results.append(
                    f"Title: {result.get('title')}\n"
                    f"URL: {result.get('url')}\n"
                    f"Content: {result.get('content')}"
                )
                
            if not results:
                return f"No results found on Tavily for query: '{query}'."
                
            return "\n\n".join(results)
            
        except requests.exceptions.RequestException as e:
            return f"Tavily search API request failed: {str(e)}"
        except Exception as e:
            return f"An unexpected error occurred during Tavily search: {str(e)}"
