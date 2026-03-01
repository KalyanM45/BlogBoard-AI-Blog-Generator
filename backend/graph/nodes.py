"""
nodes.py
Five LangGraph node functions that implement the BlogBoard article-generation pipeline.

Node execution order:
    1. consolidate_schedule   – merge input JSONs → schedule.json
    2. get_domain_topic       – pick today's domain + topic from the schedule
    3. llm_generate           – one Groq call → title, description, tags, content
    4. save_markdown          – write the .md file to frontend/blogs/{domain}/
    5. update_articles_json   – upsert entry in frontend/blogs/{domain}/articles.json
"""

from __future__ import annotations

import json
import math
import os
import re

from graph.state import BlogState
from config.paths   import INPUT_DIR, SCHEDULE_FILE, BLOGS_DIR, PROMPT_FILE
from config.model   import MODEL, TEMPERATURE, MAX_TOKENS, WORDS_PER_MINUTE
from config.domains import DOMAIN_MAP, CATEGORY_META


# ─────────────────────────────────────────────────────────────────────────────
#  Private utilities
# ─────────────────────────────────────────────────────────────────────────────

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


# ─────────────────────────────────────────────────────────────────────────────
#  Node 1 – Consolidate Schedule
# ─────────────────────────────────────────────────────────────────────────────

def consolidate_schedule(state: BlogState) -> BlogState:
    """
    Reads all input JSONs from backend/data/input/, merges them into
    backend/schedule.json, and loads the result into state["schedule"].
    """
    print("\n[Node 1] Consolidating schedule from input JSONs…")

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

            if not date or not topic:
                print(f"  [WARN]  Skipping entry with missing date/topic in {filename}: {entry}")
                continue

            if date in schedule:
                conflicts.append((date, schedule[date]["domain"], domain))
                print(
                    f"  [WARN]  Date conflict: {date} already assigned to "
                    f"{schedule[date]['domain']}, overwriting with {domain}"
                )

            schedule[date] = {"domain": domain, "topic": topic}
            total += 1

    # Sort chronologically
    schedule = dict(sorted(schedule.items()))

    # Persist to schedule.json
    with open(SCHEDULE_FILE, "w", encoding="utf-8") as f:
        json.dump(schedule, f, indent=2, ensure_ascii=False)

    print(f"  ✅ schedule.json written — {total} entries, {len(schedule)} dates, {len(conflicts)} conflict(s)")
    return {**state, "schedule": schedule}


# ─────────────────────────────────────────────────────────────────────────────
#  Node 2 – Get Domain & Topic
# ─────────────────────────────────────────────────────────────────────────────

def get_domain_topic(state: BlogState) -> BlogState:
    """
    Looks up state["date"] in the schedule and extracts domain + topic.
    Sets state["skipped"] = True when no entry exists for the date.
    """
    print("\n[Node 2] Resolving domain and topic for date…")

    date     = state["date"]
    schedule = state["schedule"]

    if date not in schedule:
        print(f"  [INFO]  No article scheduled for {date}. (Skipping generation.)")
        return {**state, "skipped": True}

    entry  = schedule[date]
    domain = entry["domain"]
    topic  = entry["topic"]
    label  = CATEGORY_META.get(domain, {}).get("label", domain)

    print(f"  Date   : {date}")
    print(f"  Domain : {domain}  ({label})")
    print(f"  Topic  : {topic}")

    return {
        **state,
        "domain":  domain,
        "topic":   topic,
        "skipped": False,
    }


# ─────────────────────────────────────────────────────────────────────────────
#  Node 3 – LLM Generate
# ─────────────────────────────────────────────────────────────────────────────

def llm_generate(state: BlogState) -> BlogState:
    """
    Calls Groq with the exhaustive blog_generation prompt and parses the
    returned JSON into title, description, tags, slug, content, and read_time.
    In dry_run mode, returns placeholder values instead.
    """
    print("\n[Node 3] Running LLM generation…")

    domain    = state["domain"]
    topic     = state["topic"]
    cat_label = CATEGORY_META.get(domain, {}).get("label", domain)

    # ── Dry-run shortcut ──────────────────────────────────────────────────────
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

    # ── Load prompt template ──────────────────────────────────────────────────
    if not PROMPT_FILE.exists():
        raise FileNotFoundError(f"Prompt file not found: {PROMPT_FILE}")

    with open(PROMPT_FILE, "r", encoding="utf-8") as f:
        prompt_template = f.read()

    draft_title = topic[:70]
    prompt = prompt_template.format(
        cat_label=cat_label,
        topic=topic,
        title=draft_title,
    )

    print(f"  ⏳ Calling Groq ({MODEL})…")
    client   = _get_groq_client()
    response = client.chat.completions.create(
        model=MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=TEMPERATURE,
        max_tokens=MAX_TOKENS,
    )

    raw = response.choices[0].message.content.strip()

    title       = draft_title
    description = raw[:150].replace('\n', ' ') + "..."
    tags        = [domain, "tutorial"]
    content     = raw
    slug        = _slugify(title)
    rt          = _read_time(content)

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


# ─────────────────────────────────────────────────────────────────────────────
#  Node 4 – Save Markdown
# ─────────────────────────────────────────────────────────────────────────────

def save_markdown(state: BlogState) -> BlogState:
    """
    Writes the generated article content to:
        frontend/blogs/{domain}/{slug}.md
    Skips the write in dry_run mode.
    """
    print("\n[Node 4] Saving markdown file…")

    domain  = state["domain"]
    slug    = state["slug"]
    content = state["content"]

    domain_dir  = BLOGS_DIR / domain
    md_filename = f"{slug}.md"
    md_path     = domain_dir / md_filename

    if state.get("dry_run"):
        print(f"  [DRY RUN] Would write: frontend/blogs/{domain}/{md_filename}")
        return {**state, "md_path": str(md_path)}

    domain_dir.mkdir(parents=True, exist_ok=True)
    with open(md_path, "w", encoding="utf-8") as f:
        f.write(content)

    print(f"  ✅ Saved: {md_path}")
    return {**state, "md_path": str(md_path)}


# ─────────────────────────────────────────────────────────────────────────────
#  Node 5 – Update articles.json
# ─────────────────────────────────────────────────────────────────────────────

def update_articles_json(state: BlogState) -> BlogState:
    """
    Upserts the article metadata into frontend/blogs/{domain}/articles.json,
    sorts entries by date descending, and saves.
    Skips the write in dry_run mode.
    """
    print("\n[Node 5] Updating articles.json…")

    domain      = state["domain"]
    slug        = state["slug"]
    title       = state["title"]
    description = state["description"]
    tags        = state["tags"]
    read_time   = state["read_time"]
    date        = state["date"]
    md_filename = f"{slug}.md"
    md_relative = f"blogs/{domain}/{md_filename}"

    articles_path = BLOGS_DIR / domain / "articles.json"

    if state.get("dry_run"):
        print(f"  [DRY RUN] Would update: frontend/blogs/{domain}/articles.json")
        return state

    # Load existing articles (gracefully handle missing file)
    articles: list[dict] = []
    if articles_path.exists():
        with open(articles_path, "r", encoding="utf-8") as f:
            articles = json.load(f)

    # Remove any existing entry for this same slug to avoid duplicates
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

    articles_path.parent.mkdir(parents=True, exist_ok=True)
    with open(articles_path, "w", encoding="utf-8") as f:
        json.dump(articles_sorted, f, indent=2, ensure_ascii=False)

    print(f"  ✅ articles.json updated: frontend/blogs/{domain}/articles.json  ({len(articles_sorted)} entries)")
    return state
