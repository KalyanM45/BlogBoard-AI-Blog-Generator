"""
config/model.py
LLM model configuration for the BlogBoard generation pipeline.
"""

# ── Groq model identifier ─────────────────────────────────────────────────────
MODEL: str = "llama-3.3-70b-versatile"

# ── Generation parameters ─────────────────────────────────────────────────────
TEMPERATURE: float = 0.65
MAX_TOKENS:  int   = 4096

# ── Approximate words-per-minute used for read-time estimation ────────────────
WORDS_PER_MINUTE: int = 200
