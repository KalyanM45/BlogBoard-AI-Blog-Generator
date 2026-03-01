#!/usr/bin/env python3
"""
generate_article.py
Reads today's schedule entry from backend/schedule.json, generates a full AI blog
article using Groq (llama-3.3-70b-versatile), and saves it to frontend/blogs/{domain}/.

Usage:
    python backend/scripts/generate_article.py
    python backend/scripts/generate_article.py --date 2026-03-07
    python backend/scripts/generate_article.py --dry-run
"""

import json
import math
import os
import re
import sys
import argparse
from datetime import datetime, timezone, timedelta
from pathlib import Path

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

# ── Paths ─────────────────────────────────────────────────
BACKEND_DIR   = Path(__file__).parent.parent           # BlogBoard/backend/
ROOT          = BACKEND_DIR.parent                     # BlogBoard/
SCHEDULE_FILE = BACKEND_DIR / "schedule.json"
BLOGS_DIR     = ROOT / "frontend" / "blogs"

# ── Groq model ────────────────────────────────────────────
MODEL = "llama-3.3-70b-versatile"

# ── Category metadata ──────────────────────────────────────
CATEGORY_META = {
    "ml":         {"label": "Machine Learning",            "shortLabel": "ML"},
    "dl":         {"label": "Deep Learning",               "shortLabel": "DL"},
    "nlp":        {"label": "Natural Language Processing", "shortLabel": "NLP"},
    "cv":         {"label": "Computer Vision",             "shortLabel": "CV"},
    "genai":      {"label": "Generative AI",               "shortLabel": "Gen AI"},
    "ainews":     {"label": "AI News",                     "shortLabel": "AI News"},
    "statistics": {"label": "Statistics for AI",           "shortLabel": "Stats"},
}


# ─────────────────────────────────────────────────────────
#  Helpers
# ─────────────────────────────────────────────────────────

def today_ist() -> str:
    ist = timezone(timedelta(hours=5, minutes=30))
    return datetime.now(ist).strftime("%Y-%m-%d")


def slugify(text: str) -> str:
    text = text.lower()
    text = re.sub(r"[^\w\s-]", "", text)
    text = re.sub(r"[\s_-]+", "-", text)
    return text.strip("-")[:80]


def read_time(text: str) -> str:
    return f"{math.ceil(len(text.split()) / 200)} min"


def load_schedule() -> dict:
    if not SCHEDULE_FILE.exists():
        print("[ERROR] backend/schedule.json not found. Run: python backend/scripts/build_schedule.py")
        sys.exit(1)
    with open(SCHEDULE_FILE, "r", encoding="utf-8") as f:
        return json.load(f)


def load_articles_json(domain: str) -> list:
    path = BLOGS_DIR / domain / "articles.json"
    if path.exists():
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    return []


def save_articles_json(domain: str, articles: list):
    path = BLOGS_DIR / domain / "articles.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    articles_sorted = sorted(articles, key=lambda x: x["date"], reverse=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(articles_sorted, f, indent=2, ensure_ascii=False)


# ─────────────────────────────────────────────────────────
#  Groq Generation
# ─────────────────────────────────────────────────────────

def get_groq_client():
    api_key = os.environ.get("GROQ_API_KEY", "").strip()
    if not api_key:
        print("[ERROR] GROQ_API_KEY not set. Add it to .env or export it.")
        sys.exit(1)
    try:
        from groq import Groq
        return Groq(api_key=api_key)
    except ImportError:
        print("[ERROR] groq not installed. Run: pip install -r requirements.txt")
        sys.exit(1)


def generate_metadata(client, domain: str, topic: str) -> dict:
    cat_label = CATEGORY_META.get(domain, {}).get("label", domain)
    prompt = f"""You are an expert AI/ML technical blogger specializing in {cat_label}.

Given this article topic: "{topic}"

Generate the following in valid JSON (no markdown, no extra text):
{{
  "title": "A catchy, SEO-friendly article title (max 70 chars)",
  "description": "A compelling 1-2 sentence meta description (max 160 chars)",
  "tags": ["tag1", "tag2", "tag3", "tag4", "tag5"]
}}

Rules:
- Title must include the core technical keywords
- Description must highlight reader value
- Tags must be lowercase, hyphenated technical terms
- Output ONLY the JSON object"""

    response = client.chat.completions.create(
        model=MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.7,
        max_tokens=300,
    )
    raw = response.choices[0].message.content.strip()
    raw = re.sub(r"^```json\s*", "", raw, flags=re.MULTILINE)
    raw = re.sub(r"```\s*$", "", raw, flags=re.MULTILINE)
    return json.loads(raw.strip())


def generate_article_content(client, domain: str, topic: str, title: str) -> str:
    cat_label = CATEGORY_META.get(domain, {}).get("label", domain)
    prompt = f"""You are an expert AI/ML technical blogger specializing in {cat_label}.

Write a comprehensive, publication-ready technical blog post:
- Title: {title}
- Topic: {topic}
- Category: {cat_label}

Requirements:
1. Start DIRECTLY with the article content — DO NOT include the title as a heading
2. Write 1000-1500 words of high-quality technical content
3. Use clear markdown formatting (##, ###, code blocks with language tags, tables, bold for key terms)
4. Structure: intro → core concepts → code example → real-world applications → conclusion
5. Tone: Expert but approachable
6. Output ONLY the markdown content

Begin the article now:"""

    response = client.chat.completions.create(
        model=MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.6,
        max_tokens=3000,
    )
    return response.choices[0].message.content.strip()


# ─────────────────────────────────────────────────────────
#  Main Pipeline
# ─────────────────────────────────────────────────────────

def run(date_str: str, dry_run: bool = False):
    print(f"\n{'='*55}")
    print(f"  BlogBoard Article Generator")
    print(f"  Date    : {date_str}")
    print(f"  Dry run : {dry_run}")
    print(f"{'='*55}\n")

    schedule = load_schedule()

    if date_str not in schedule:
        print(f"[INFO] No article scheduled for {date_str}.")
        print(f"       (This is expected on Sundays or unscheduled dates.)")
        sys.exit(0)

    entry     = schedule[date_str]
    domain    = entry["domain"]
    topic     = entry["topic"]
    cat_label = CATEGORY_META.get(domain, {}).get("label", domain)

    print(f"  Domain  : {domain} ({cat_label})")
    print(f"  Topic   : {topic}\n")

    if dry_run:
        print("[DRY RUN] Skipping Groq generation and file writes.")
        print(f"[DRY RUN] Would generate: frontend/blogs/{domain}/<slug>.md")
        print(f"[DRY RUN] Would update  : frontend/blogs/{domain}/articles.json")
        return

    print("⏳ Generating metadata (title, description, tags)...")
    client = get_groq_client()
    meta        = generate_metadata(client, domain, topic)
    title       = meta["title"]
    description = meta["description"]
    tags        = meta["tags"]
    slug        = slugify(title)

    print(f"  Title       : {title}")
    print(f"  Description : {description}")
    print(f"  Tags        : {', '.join(tags)}")
    print(f"  Slug        : {slug}\n")

    print("⏳ Generating article content...")
    content = generate_article_content(client, domain, topic, title)
    rt      = read_time(content)
    print(f"  Read time   : {rt}")
    print(f"  Word count  : {len(content.split())}\n")

    # Save markdown
    domain_dir  = BLOGS_DIR / domain
    domain_dir.mkdir(parents=True, exist_ok=True)
    md_filename = f"{slug}.md"
    md_path     = domain_dir / md_filename
    md_relative = f"blogs/{domain}/{md_filename}"

    with open(md_path, "w", encoding="utf-8") as f:
        f.write(content)
    print(f"✅ Article saved : {md_path}")

    # Update articles.json
    articles = load_articles_json(domain)
    articles = [a for a in articles if a.get("id") != md_relative]
    articles.append({
        "id":          md_relative,
        "category":    domain,
        "title":       title,
        "description": description,
        "date":        date_str,
        "tags":        tags,
        "readTime":    rt,
        "file":        md_relative,
    })
    save_articles_json(domain, articles)
    print(f"✅ articles.json : frontend/blogs/{domain}/articles.json")
    print(f"\n🎉 Done! Article for {date_str} generated successfully.\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="BlogBoard Article Generator (Groq)")
    parser.add_argument("--date", type=str, default=None,
                        help="Date to generate (YYYY-MM-DD). Defaults to today in IST.")
    parser.add_argument("--dry-run", action="store_true",
                        help="Preview only — no file writes or Groq calls.")
    args = parser.parse_args()
    run(args.date or today_ist(), dry_run=args.dry_run)
