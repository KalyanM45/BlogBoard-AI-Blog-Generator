"""
config/domains.py
Domain and category mapping constants for the BlogBoard pipeline.

DOMAIN_MAP      — maps input JSON filenames → short domain keys
CATEGORY_META   — maps short domain keys   → human-readable labels
"""

from typing import Dict

# ── Input filename → domain key ───────────────────────────────────────────────
DOMAIN_MAP: Dict[str, str] = {
    "machine_learning.json":            "ml",
    "deep_learning.json":               "dl",
    "natural_language_processing.json": "nlp",
    "computer_vision.json":             "cv",
    "generative_ai.json":               "genai",
    "statistics.json":                  "statistics",
}

# ── Domain key → display labels ───────────────────────────────────────────────
CATEGORY_META: Dict[str, Dict[str, str]] = {
    "ml":         {"label": "Machine Learning",            "shortLabel": "ML"},
    "dl":         {"label": "Deep Learning",               "shortLabel": "DL"},
    "nlp":        {"label": "Natural Language Processing", "shortLabel": "NLP"},
    "cv":         {"label": "Computer Vision",             "shortLabel": "CV"},
    "genai":      {"label": "Generative AI",               "shortLabel": "Gen AI"},
    "ainews":     {"label": "AI News",                     "shortLabel": "AI News"},
    "statistics": {"label": "Statistics for AI",           "shortLabel": "Stats"},
}
