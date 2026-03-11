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

def consolidate_schedule(state: BlogState) -> BlogState:

    if not INPUT_DIR.exists():
        raise FileNotFoundError(f"Input directory not found: {INPUT_DIR}")

    schedule: dict[str, dict] = {}
    total = 0
    conflicts: list[tuple] = []

    for filename, domain in DOMAIN_MAP.items():
        filepath = INPUT_DIR / filename
        if not filepath.exists():
            print(f"  [WARN]  Skipping missing file: {filename}")
            continue

        with open(filepath, "r", encoding="utf-8") as f:
            entries = json.load(f)

        for entry in entries:
            date  = entry.get("date", "").strip()
            topic = entry.get("topic", "").strip()
            subtopics = entry.get("subtopics", "").strip()

            if not date or not topic:
                print(f"  [WARN]  Skipping entry with missing date/topic in {filename}: {entry}")
                continue

            if date in schedule:
                conflicts.append((date, schedule[date]["domain"], domain))
                print(
                    f"  [WARN]  Date conflict: {date} already assigned to "
                    f"{schedule[date]['domain']}, overwriting with {domain}"
                )

            schedule_entry = {"domain": domain, "topic": topic}
            if subtopics:
                schedule_entry["subtopics"] = subtopics

            schedule[date] = schedule_entry
            total += 1


    schedule = dict(sorted(schedule.items()))
    with open(SCHEDULE_FILE, "w", encoding="utf-8") as f:
        json.dump(schedule, f, indent=2, ensure_ascii=False)

    print(f"  ✅ schedule.json written — {total} entries, {len(schedule)} dates, {len(conflicts)} conflict(s)")
    return {**state, "schedule": schedule}



def get_domain_topic(state: BlogState) -> BlogState:

    date, schedule = state["date"], state["schedule"]

    if date not in schedule:
        print(f"  [INFO]  No article scheduled for {date}. (Skipping generation.)")
        return {**state, "skipped": True}

    entry = schedule[date]
    domain, topic = entry["domain"], entry["topic"]
    subtopics = entry.get("subtopics", "")
    label = CATEGORY_META.get(domain, {}).get("label", domain)

    print(f"  Date      : {date}")
    print(f"  Domain    : {domain}  ({label})")
    print(f"  Topic     : {topic}")
    if subtopics:
        print(f"  Subtopics : {subtopics}")

    return {**state, "domain": domain, "topic": topic, "subtopics": subtopics, "skipped": False}


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
