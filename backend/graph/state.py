from typing import TypedDict


class BlogState(TypedDict, total=False):
    date: str
    dry_run: bool
    schedule: dict
    domain: str
    topic: str
    subtopics: str
    skipped: bool
    title: str
    description: str
    tags: list
    slug: str
    content: str
    read_time: str
    md_path: str
    news_data: str