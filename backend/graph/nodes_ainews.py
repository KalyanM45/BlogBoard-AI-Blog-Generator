import json
import os
from datetime import datetime, timedelta
import requests

from graph.state import BlogState
from graph.nodes import _slugify, _read_time, _get_groq_client
from config.config import MODEL, PROMPT_DIRECTORY, BLOGS_DIR


def fetch_news(state: BlogState) -> BlogState:
    date_str = state["date"]
    current_date = datetime.strptime(date_str, "%Y-%m-%d")
    
    days_since_sunday = (current_date.weekday() + 1) % 7
    sunday = current_date - timedelta(days=days_since_sunday)
    saturday = sunday + timedelta(days=6)
    
    start_date_str = sunday.strftime("%Y-%m-%d")
    end_date_str = saturday.strftime("%Y-%m-%d")
    
    print(f"  [INFO] Fetching AI news from {start_date_str} to {end_date_str}...")
    
    news_items = []
    
    story_images = []

    # 1. Tavily API
    tavily_api_key = os.environ.get("TAVILY_API_KEY", "").strip()
    if tavily_api_key:
        try:
            print("  ⏳ Fetching from Tavily...")
            response = requests.post(
                "https://api.tavily.com/search",
                json={
                    "api_key": tavily_api_key,
                    "query": "artificial intelligence OR AI news",
                    "topic": "news",
                    "days": 7,
                    "max_results": 3,
                    "include_raw_content": False,
                    "include_images": True
                },
                timeout=10
            )
            response.raise_for_status()
            data = response.json()
            images = data.get("images", [])
            for i, result in enumerate(data.get("results", [])):
                img = images[i] if i < len(images) else ""
                if img:
                    story_images.append(img)
                news_items.append(f"Tavily Article {i+1}:\nTitle: {result.get('title')}\nURL: {result.get('url')}\nImage URL: {img}\nContent: {result.get('content')}\n")
        except Exception as e:
            print(f"  [WARN] Failed to fetch from Tavily: {e}")
    else:
        print("  [WARN] TAVILY_API_KEY not set, skipping Tavily.")

    # 2. Guardian API
    guardian_api_key = os.environ.get("GUARDIAN_API_KEY", "").strip()
    if guardian_api_key:
        try:
            print("  ⏳ Fetching from Guardian...")
            response = requests.get(
                "https://content.guardianapis.com/search",
                params={
                    "api-key": guardian_api_key,
                    "q": '("artificial intelligence" OR " AI ")',
                    "from-date": start_date_str,
                    "to-date": end_date_str,
                    "order-by": "relevance",
                    "page-size": 3,
                    "show-fields": "headline,trailText,bodyText,thumbnail"
                },
                timeout=10
            )
            response.raise_for_status()
            data = response.json()
            results = data.get("response", {}).get("results", [])
            for i, result in enumerate(results):
                fields = result.get("fields", {})
                thumbnail = fields.get('thumbnail', '')
                if thumbnail:
                    story_images.append(thumbnail)
                news_items.append(
                    f"Guardian Article {i+1}:\n"
                    f"Title: {fields.get('headline')}\n"
                    f"URL: {result.get('webUrl')}\n"
                    f"Image URL: {thumbnail}\n"
                    f"Excerpt: {fields.get('trailText')}\n"
                    f"Content: {fields.get('bodyText', '')[:1000]}...\n"
                )
        except Exception as e:
            print(f"  [WARN] Failed to fetch from Guardian: {e}")
    else:
        print("  [WARN] GUARDIAN_API_KEY not set, skipping Guardian.")

    print(f"  [INFO] Adding unsplash images to news stories")
    
    # Let's get a main image for the AI generic topic
    main_image_url = ""
    unsplash_api_key = os.environ.get("UNSPLASH_API_KEY", "").strip()
    if unsplash_api_key:
        try:
             response = requests.get(
                "https://api.unsplash.com/photos/random",
                params={"query": "artificial intelligence", "orientation": "landscape"},
                headers={"Authorization": f"Client-ID {unsplash_api_key}"},
                timeout=10
             )
             response.raise_for_status()
             data = response.json()
             if data and isinstance(data, dict):
                 main_image_url = data.get("urls", {}).get("regular", "")
        except Exception as e:
             print(f"  [WARN] Failed to fetch Unsplash image: {e}")
    
    if not main_image_url:
        # Fallback random image
        main_image_url = "https://images.unsplash.com/photo-1677442136019-21780ecad995?q=80&w=1000&auto=format&fit=crop"

    if not news_items:
        print("  [WARN] No news fetched. Generating a generic AI news placeholder.")
        news_data = "No specific news found. Provide a general overview of AI trends this week."
    else:
        news_data = "\n\n".join(news_items)

    # Count existing articles
    articles_path = BLOGS_DIR / "ainews" / "articles.json"
    article_count = 1
    if articles_path.exists() and articles_path.stat().st_size > 0:
        try:
            with open(articles_path, "r", encoding="utf-8") as f:
                articles = json.load(f)
                article_count = len(articles) + 1
        except Exception:
            pass

    title = f"The Week in AI #{article_count}"
    
    # Optional formatting for story images to pass to LLM
    story_image_links_if_any = "\n".join(story_images) if story_images else "No additional images available."

    state_update = {
        **state,
        "domain": "ainews",
        "topic": title,
        "title": title,
        "news_data": news_data,
        "skipped": False,
        "tags": ["ainews", "artificial intelligence", "weekly roundup"]
    }
    # Pass additional fields not strictly typed in BlogState but used by the next node
    state_update["header_image_url"] = main_image_url
    state_update["story_image_links_if_any"] = story_image_links_if_any
    return state_update

def llm_generate_ainews(state: BlogState) -> BlogState:
    domain, title = state["domain"], state["title"]
    news_data = state.get("news_data", "")
    header_image_url = state.get("header_image_url", "")
    story_image_links_if_any = state.get("story_image_links_if_any", "")
    
    slug = _slugify(title)
    
    if state.get("dry_run"):
        placeholder_content = f"# {title}\n\nDry run — no LLM call made. News context:\n{news_data[:200]}..."
        print("  [DRY RUN] Skipping Groq call for AI News.")
        return {
            **state,
            "description": "A weekly roundup of the most important AI news.",
            "slug": slug,
            "content": placeholder_content,
            "read_time": "1 min",
        }

    print(f"  ⏳ Generating article content using fetched news...")
    
    prompt_path = PROMPT_DIRECTORY / "ainews_generation.txt"
    if not prompt_path.exists():
        raise FileNotFoundError(f"AI news prompt not found: {prompt_path}")

    with open(prompt_path, "r", encoding="utf-8") as f:
        base_prompt = f.read()

    base_prompt = base_prompt.replace("{title}", title)
    base_prompt = base_prompt.replace("{header_image_url}", header_image_url)
    base_prompt = base_prompt.replace("{news_context}", news_data)
    base_prompt = base_prompt.replace("{story_image_links_if_any}", story_image_links_if_any)

    client = _get_groq_client()
    blog_res = client.chat.completions.create(
        model=MODEL,
        messages=[{"role": "user", "content": base_prompt}],
        temperature=0.6,
        max_tokens=3000,
    )

    content = blog_res.choices[0].message.content.strip()
    rt = _read_time(content)
    
    # Try getting a description
    description = "A weekly roundup of the most important advancements and news in Artificial Intelligence."
    try:
        desc_res = client.chat.completions.create(
            model=MODEL,
            messages=[{"role": "user", "content": f"Provide a single brief sentence describing this article:\n\n{title}\n\n{content[:500]}..."}],
            temperature=0.5,
            max_tokens=100
        )
        description = desc_res.choices[0].message.content.strip().replace('"', '')
    except Exception as e:
        print(f"  [WARN] Failed to generate description: {e}")

    print(f"  Title       : {title}")
    print(f"  Description : {description}")
    print(f"  Tags        : {', '.join(state['tags'])}")
    print(f"  Slug        : {slug}")
    print(f"  Word count  : {len(content.split())}")
    print(f"  Read time   : {rt}")

    return {
        **state,
        "description": description,
        "slug": slug,
        "content": content,
        "read_time": rt,
    }
