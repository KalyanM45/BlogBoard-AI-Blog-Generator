import json
import re
from datetime import datetime
from blogboard.graph.state import BlogState
from blogboard.services.llm import LLMAgentService
from blogboard.services.storage import R2StorageService
from blogboard.services.prompt_manager import prompt_manager
from .prompts import VALIDATOR_PROMPT

def validator_node(state: BlogState) -> BlogState:
    print("  => [ValidatorAgent] Running...")
    
    if state.get("dry_run"):
        print("  [DRY RUN] Simulating Approval and Metadata Gen.")
        return {
            **state, 
            "revision_needed": False,
            "title": "Dry Run Generated Title",
            "slug": "dry-run-generated",
            "md_path": "r2://dry-run",
            "description": "Dry run generic description.",
        }

    current_revision = state.get("revision_count", 0)
    topic = state.get("topic")
    content = state.get("content", "")
    domain = state.get("domain")
    date = state.get("date", datetime.now().strftime("%Y-%m-%d"))

    prompt = prompt_manager.get_prompt(
        prompt_name="Validator_Prompt",
        fallback_prompt=VALIDATOR_PROMPT,
        topic=topic,
        content=content
    )
    
    llm_service = LLMAgentService(temperature=0.1) 
    res = llm_service.llm.invoke(prompt)
    
    raw = res.content.strip()
    raw = re.sub(r"^```json\s*", "", raw, flags=re.MULTILINE)
    raw = re.sub(r"```\s*$", "", raw, flags=re.MULTILINE)
    
    try:
        data = json.loads(raw.strip())
        approved = data.get("approved", True)
        feedback = data.get("feedback", "")
        title = data.get("title", topic)
        description = data.get("description", "A blog post about " + topic)
        slug_value = data.get("slug", title.lower().replace(" ", "-"))
    except json.JSONDecodeError:
        print("  [WARN] Validator failed to return JSON. Forcing approval fallback.")
        approved = True
        feedback = ""
        title = topic[:70] if topic else "fallback"
        description = "A blog post about " + title
        slug_value = title.lower().replace(" ", "-")
        
    if not approved and current_revision >= 3:
        print("  [WARN] Max revisions reached. Forcing approval.")
        approved = True
        
    revision_needed = not approved
    
    if revision_needed:
        print(f"  [AGENT] Draft REJECTED. Feedback: {feedback}")
        return {
            **state,
            "revision_needed": True,
            "validator_feedback": feedback,
            "revision_count": current_revision + 1
        }
        
    print(f"  [AGENT] Draft APPROVED! Generating Metadata and Saving to R2...")
    
    # Save to R2
    slug_value = re.sub(r"[^\w\s-]", "", slug_value).strip("-")
    md_relative = f"blogs/{domain}/{slug_value}.md"
    storage = R2StorageService()
    
    storage.put_object(md_relative, content, content_type="text/markdown")
    articles = storage.get_articles_json(domain)
    articles = [a for a in articles if a.get("id") != md_relative]
    
    articles.append({
        "id": md_relative,
        "category": domain,
        "topic": topic,
        "subtopics": state.get("subtopics", ""),
        "title": title,
        "description": description,
        "date": date,
        "tags": [domain],
        "readTime": state.get("read_time", "5 min"),
        "file": md_relative,
    })
    
    articles = sorted(articles, key=lambda x: x["date"], reverse=True)
    storage.save_articles_json(domain, articles)
    
    return {
        **state,
        "revision_needed": False,
        "title": title,
        "description": description,
        "slug": slug_value,
        "md_path": f"r2://{md_relative}"
    }
