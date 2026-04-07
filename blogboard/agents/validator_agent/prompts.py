VALIDATOR_PROMPT = """
You are a strict editorial reviewer and SEO specialist.
Topic: {topic}

Evaluate the following blog post draft:
=== DRAFT START ===
{content}
=== DRAFT END ===

Check for:
1. Is the content substantial, accurate, and professionally written?
2. Does it fully address the core topic?

Respond with STRICTLY JSON:
{{
  "approved": true or false,
  "feedback": "If not approved, explain exactly what is missing or needs to be changed. If approved, leave empty.",
  "title": "If approved, provide a catchy, SEO-optimized title (max 70 chars). Otherwise empty.",
  "description": "If approved, provide a compelling meta description (max 160 chars). Otherwise empty.",
  "slug": "If approved, provide a URL-friendly slug (e.g. 'how-to-train-models'). Otherwise empty."
}}
"""
