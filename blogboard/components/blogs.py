from blogboard.utils.llm_utils import GroqClient
from blogboard.schema.schema import TitleResponse
from blogboard.graph.state import BlogState
from blogboard.database.db import get_recent_blog_topics

class BlogNodes:
    def __init__(self):
        self.llm = GroqClient()
    
    def TopicSelection(self, state: BlogState) -> BlogState:
        # 1. Retrieve the last 3 blog topics for the current domain from R2
        recent_topics = get_recent_blog_topics(state["domain"])
        
        # 2. Generate the next topic using the LLM with context
        response = self.llm.generate_content(
            prompt_name="topic_selection",
            schema=TitleResponse,
            **{
                "DOMAIN": state["domain"], # Fixed casing to match prompt template
                "RECENT_TOPICS": recent_topics
            }
        )
        state["topic"] = response.title
        state["subtopics"] = response.subtopics
        return state