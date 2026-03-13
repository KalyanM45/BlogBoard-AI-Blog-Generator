from typing import List
from pydantic import BaseModel, Field

class TitleResponse(BaseModel):
    title: str = Field(description="The catchy, SEO-friendly title of the blog post")
    subtopics: str = Field(description="A String of 4 to 6 specific subtopics to be covered separated by commas")