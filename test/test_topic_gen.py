import os
import sys
from dotenv import load_dotenv

# Load environment variables from .env
load_dotenv()

# Add the current directory to sys.path so we can import 'blogboard'
sys.path.append(os.getcwd())

from blogboard.components.blogs import BlogNodes
from blogboard.utils.domain_utils import get_current_day_domain

def test_topic_generation():
    print("🚀 Starting Autonomous Topic Generation Test...")
    
    # 1. Initialize the Nodes (contains the LLM client)
    try:
        nodes = BlogNodes()
    except Exception as e:
        print(f"❌ Failed to initialize BlogNodes: {e}")
        return

    # 2. Determine today's domain automatically
    domain = get_current_day_domain()
    print(f"📅 Today's Domain: {domain}")
    
    # 3. Setup Mock State
    # Note: We only need the keys used by TopicSelection
    state = {
        "domain": domain,
        "topic": "",
        "subtopics": "",
        # Adding mandatory fields from BlogState TypedDict to avoid runtime errors
        "date": "2024-03-13",
        "dry_run": True,
        "skipped": False,
        "recent_blogs": [],
        "title": "",
        "description": "",
        "tags": [],
        "slug": "",
        "content": "",
        "read_time": "",
        "md_path": "",
        "news_data": "",
        "schedule": {}
    }
    
    print("📡 Fetching history from R2 and generating topic (this may take a few seconds)...")
    
    try:
        # 4. Run the TopicSelection node
        updated_state = nodes.TopicSelection(state)
        
        print("\n✨ SUCCESS! ✨")
        print("-" * 30)
        print(f"Final Selection for {domain}:")
        print(f"Topic: {updated_state['topic']}")
        print(f"Subtopics: {updated_state['subtopics']}")
        print("-" * 30)
        
    except Exception as e:
        print(f"\n❌ Test Failed during LLM/R2 execution: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_topic_generation()
