import json
import math
import os
import re

from graph.state import BlogState
from config.config import (
    DOMAIN_MAP, CATEGORY_META, MODEL, TEMPERATURE, MAX_TOKENS, WORDS_PER_MINUTE,
    INPUT_DIR, SCHEDULE_FILE, PROMPT_DIRECTORY, BLOGS_DIR, BACKEND_DIR
)


def _slugify(text: str) -> str:
    text = text.lower()
    text = re.sub(r"[^\w\s-]", "", text)
    text = re.sub(r"[\s_-]+", "-", text)
    return text.strip("-")[:80]


def _read_time(text: str) -> str:
    return f"{math.ceil(len(text.split()) / WORDS_PER_MINUTE)} min"


def _get_groq_client():
    api_key = os.environ.get("GROQ_API_KEY", "").strip()
    if not api_key:
        raise EnvironmentError(
            "[ERROR] GROQ_API_KEY not set. Add it to .env or export it."
        )
    try:
        from groq import Groq
        return Groq(api_key=api_key)
    except ImportError:
        raise ImportError(
            "[ERROR] groq package not installed. Run: pip install -r backend/requirements.txt"
        )

from storage import get_recent_history, get_all_domains_last_updated
import random

def autonomous_topic_selection(state: BlogState) -> BlogState:
    """Autonomously selects a domain and topic based on recent R2 history, avoiding repetition."""
    print("  => Autonomous Topic Agent running...")

    # 1. Pick a domain that needs an article (e.g., oldest updated)
    domain_dates = get_all_domains_last_updated()
    
    # Sort domains by oldest first (or 'Never' first).
    # Since dates are ISO YYYY-MM-DD, standard string sort works perfectly.
    sorted_domains = sorted(domain_dates.items(), key=lambda item: item[1])
    target_domain = sorted_domains[0][0] # The one least recently updated
    cat_label = CATEGORY_META.get(target_domain, {}).get("label", target_domain)
    print(f"  [AGENT] Selected domain: {target_domain} (Last updated: {domain_dates[target_domain]})")

    # 2. Fetch the last 3 articles for this domain to use as history block
    recent_history = get_recent_history(target_domain, limit=3)
    
    history_str = "No recent history found."
    if recent_history:
        history_str = "\n---\n".join([
            f"Title: {item['title']}\nTopic: {item['topic']}\nSubtopics: {item['subtopics']}"
            for item in recent_history
        ])
    
    # 3. Call Groq to pick a new topic
    prompt_path = BACKEND_DIR / "prompts" / "autonomous_topic_selection.txt"
    if not prompt_path.exists():
        raise FileNotFoundError(f"Missing prompt file: {prompt_path}")
        
    with open(prompt_path, "r", encoding="utf-8") as f:
        prompt_template = f.read()
        
    prompt = prompt_template.replace("{cat_label}", cat_label).replace("{history}", history_str)

    client = _get_groq_client()
    res = client.chat.completions.create(
        model=MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.8,
        max_tokens=250,
    )
    
    raw = res.choices[0].message.content.strip()
    raw = re.sub(r"^```json\s*", "", raw, flags=re.MULTILINE)
    raw = re.sub(r"```\s*$", "", raw, flags=re.MULTILINE)
    
    try:
        topic_data = json.loads(raw.strip())
        topic = topic_data.get("topic", "Advanced Concepts")
        subtopics = topic_data.get("subtopics", "")
    except json.JSONDecodeError:
        print(f"  [WARN] Failed to parse Autonomous Topic JSON. Raw response: {raw}")
        topic = "Emerging Trends in " + cat_label
        subtopics = ""

    print(f"  [AGENT] Chosen Topic: {topic}")
    print(f"  [AGENT] Subtopics   : {subtopics}")

    return {
        **state,
        "domain": target_domain,
        "topic": topic,
        "subtopics": subtopics,
        "skipped": False
    }


def llm_generate(state: BlogState) -> BlogState:

    domain, topic = state["domain"], state["topic"]

    cat_label = CATEGORY_META.get(domain, {}).get("label", domain)

    if state.get("dry_run"):
        placeholder_title   = f"[DRY RUN] {topic[:60]}"
        placeholder_content = f"# {placeholder_title}\n\nDry run — no LLM call made."
        print("  [DRY RUN] Skipping Groq call.")
        return {
            **state,
            "title":       placeholder_title,
            "description": "Dry-run placeholder description.",
            "tags":        ["dry-run", domain],
            "slug":        _slugify(placeholder_title),
            "content":     placeholder_content,
            "read_time":   "1 min",
        }

    # ── 1. Generate Metadata ──────────────────────────────────────────────────
    print(f"  ⏳ Generating metadata for {domain}…")

    metadata_prompt_path = PROMPT_DIRECTORY / "metadata_generation.txt"
    if not metadata_prompt_path.exists():
        raise FileNotFoundError(f"Metadata prompt not found: {metadata_prompt_path}")

    with open(metadata_prompt_path, "r", encoding="utf-8") as f:
        meta_prompt_template = f.read()

    meta_prompt = meta_prompt_template.replace("{cat_label}", cat_label).replace("{topic}", topic)

    client   = _get_groq_client()
    meta_res = client.chat.completions.create(
        model=MODEL,
        messages=[{"role": "user", "content": meta_prompt}],
        temperature=1.0,
        max_tokens=300,
    )

    raw_meta = meta_res.choices[0].message.content.strip()
    raw_meta = re.sub(r"^```json\s*", "", raw_meta, flags=re.MULTILINE)
    raw_meta = re.sub(r"```\s*$", "", raw_meta, flags=re.MULTILINE)
    
    try:
        meta_dict = json.loads(raw_meta.strip())
        title       = meta_dict.get("title", topic[:70])
        description = meta_dict.get("description", "")
        tags        = meta_dict.get("tags", [domain])
    except json.JSONDecodeError:
        print("  [WARN] Failed to parse JSON metadata. Using fallbacks.")
        title       = topic[:70]
        description = "A comprehensive guide on " + topic
        tags        = [domain]

    slug = _slugify(title)

    # ── 2. Generate Article Content ───────────────────────────────────────────
    print(f"  ⏳ Generating article content…")
    
    blog_prompt_path = PROMPT_DIRECTORY / "blog_generation.txt"
    if not blog_prompt_path.exists():
        raise FileNotFoundError(f"Blog prompt not found: {blog_prompt_path}")

    with open(blog_prompt_path, "r", encoding="utf-8") as f:
        base_prompt = f.read()

    subtopics = state.get("subtopics", "")
    base_prompt = base_prompt.replace("{cat_label}", cat_label)
    base_prompt = base_prompt.replace("{topic}", topic)
    base_prompt = base_prompt.replace("{title}", title)
    base_prompt = base_prompt.replace("{subtopics}", subtopics) if subtopics else base_prompt

    blog_res = client.chat.completions.create(
        model=MODEL,
        messages=[{"role": "user", "content": base_prompt}],
        temperature=0.6,
        max_tokens=3000,
    )

    content = blog_res.choices[0].message.content.strip()
    rt      = _read_time(content)

    print(f"  Title       : {title}")
    print(f"  Description : {description}")
    print(f"  Tags        : {', '.join(tags)}")
    print(f"  Slug        : {slug}")
    print(f"  Word count  : {len(content.split())}")
    print(f"  Read time   : {rt}")

    return {
        **state,
        "title":       title,
        "description": description,
        "tags":        tags,
        "slug":        slug,
        "content":     content,
        "read_time":   rt,
    }

from storage import upload_string_to_r2, get_articles_json_from_r2

def save_markdown(state: BlogState) -> BlogState:
    domain, slug, content = state["domain"], state["slug"], state["content"]
    
    object_key = f"blogs/{domain}/{slug}.md"

    if state.get("dry_run"):
        print(f"  [DRY RUN] Would upload to R2: {object_key}")
        return {**state, "md_path": f"r2://{object_key}"}

    upload_string_to_r2(content, object_key, content_type="text/markdown")

    return {**state, "md_path": f"r2://{object_key}"}


def update_articles_json(state: BlogState) -> BlogState:
    domain      = state["domain"]
    slug        = state["slug"]
    title       = state["title"]
    description = state["description"]
    tags        = state["tags"]
    read_time   = state["read_time"]
    date        = state["date"]
    
    md_relative = f"blogs/{domain}/{slug}.md"
    json_key    = f"blogs/{domain}/articles.json"

    if state.get("dry_run"):
        print(f"  [DRY RUN] Would update R2 JSON list: {json_key}")
        return state

    # Fetch existing from R2
    articles = get_articles_json_from_r2(domain)

    # Remove duplicates
    articles = [a for a in articles if a.get("id") != md_relative]

    # Append new entry
    articles.append({
        "id":          md_relative,
        "category":    domain,
        "topic":       state.get("topic", title),
        "subtopics":   state.get("subtopics", ""),
        "title":       title,
        "description": description,
        "date":        date,
        "tags":        tags,
        "readTime":    read_time,
        "file":        md_relative,
    })

    # Sort newest-first
    articles_sorted = sorted(articles, key=lambda x: x["date"], reverse=True)

    # Upload updated JSON string back to R2
    json_str = json.dumps(articles_sorted, indent=2, ensure_ascii=False)
    upload_string_to_r2(json_str, json_key, content_type="application/json")

    print(f"  ✅ R2 articles.json updated: {json_key} ({len(articles_sorted)} entries)")
    return state
