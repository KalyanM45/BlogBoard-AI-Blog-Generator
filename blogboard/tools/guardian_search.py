from typing import Type
from pydantic import BaseModel, Field
from langchain_core.tools import BaseTool

from blogboard.config.settings import app_settings
import requests
from datetime import datetime, timedelta
from typing import Type
from pydantic import BaseModel, Field
from langchain_core.tools import BaseTool


class GuardianSearchInput(BaseModel):
    """Input schema for the GuardianSearchTool."""
    query: str = Field(description="The search query to look up on The Guardian.")
    days: int = Field(default=7, description="Number of days to look back in the archives.")


class GuardianSearchTool(BaseTool):
    """
    Guardian Search Tool.
    
    This tool utilizes The Guardian's content API to fetch premium journalistic
    news articles based on a specific query.
    """
    name: str = "guardian_search"
    description: str = "Search for news articles using The Guardian API. Highly reliable for global news, journalism, and well-researched topics."
    args_schema: Type[BaseModel] = GuardianSearchInput

    # Properly configuring the Pydantic behaviour for LangChain's BaseTool
    model_config = {
        "extra": "ignore" 
    }

    def _run(self, query: str, days: int = 7) -> str:
        """
        Executes the search query against The Guardian's content API.
        
        Args:
            query (str): The search phrase.
            days (int): Number of days to look back.
            
        Returns:
            str: A formatted string containing the top search results.
        """
        api_key = app_settings.content.GUARDIAN_API_KEY
        if not api_key:
             return "Error: GUARDIAN_API_KEY is not configured in the application settings."
             
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days)
        
        start_date_str = start_date.strftime("%Y-%m-%d")
        end_date_str = end_date.strftime("%Y-%m-%d")

        try:
            response = requests.get(
                "https://content.guardianapis.com/search",
                params={
                    "api-key": api_key,
                    "q": query,
                    "from-date": start_date_str,
                    "to-date": end_date_str,
                    "order-by": "relevance",
                    "page-size": 3,
                    "show-fields": "headline,trailText,bodyText"
                },
                timeout=10
            )
            response.raise_for_status()
            data = response.json()
            
            results = data.get("response", {}).get("results", [])
            formatted_results = []
            
            for result in results:
                fields = result.get("fields", {})
                content_preview = fields.get('bodyText', '')[:1000] # Cap length
                
                formatted_results.append(
                    f"Title: {fields.get('headline')}\n"
                    f"URL: {result.get('webUrl')}\n"
                    f"Excerpt: {fields.get('trailText')}\n"
                    f"Summary: {content_preview}..."
                )
                
            if not formatted_results:
                 return f"No results found on The Guardian for query: '{query}'."
                 
            return "\n\n".join(formatted_results)
            
        except requests.exceptions.RequestException as e:
            return f"Guardian search API request failed: {str(e)}"
        except Exception as e:
            return f"An unexpected error occurred during Guardian search: {str(e)}"
