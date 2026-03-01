"""
config/paths.py
All filesystem path constants used throughout the BlogBoard backend pipeline.
"""

from pathlib import Path

# ── Root anchors ──────────────────────────────────────────────────────────────
BACKEND_DIR: Path = Path(__file__).parent.parent   # BlogBoard/backend/
ROOT_DIR:    Path = BACKEND_DIR.parent             # BlogBoard/

# ── Backend data directories ──────────────────────────────────────────────────
INPUT_DIR:     Path = BACKEND_DIR / "data" / "input"
SCHEDULE_FILE: Path = BACKEND_DIR / "schedule.json"

# ── Prompt templates ──────────────────────────────────────────────────────────
PROMPT_FILE: Path = BACKEND_DIR / "prompts" / "blog_generation.txt"

# ── Frontend output directories ───────────────────────────────────────────────
BLOGS_DIR: Path = ROOT_DIR / "frontend" / "blogs"
