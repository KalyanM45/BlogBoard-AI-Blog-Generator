from typing import TypedDict, List, Dict, Any

class BlogState(TypedDict, total=True):
    domain: str
    recent_blogs: List[str]
    topic: str
    subtopics: str
    
    skipped: bool
    title: str
    description: str
    tags: List[dict]
    slug: str
    content: str
    read_time: str
    md_path: str
    news_data: str
    date: str
    dry_run: bool
    schedule: Dict[str, Any]