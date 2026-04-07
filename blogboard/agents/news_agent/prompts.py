NEWS_GENERATION_PROMPT = """
You are a highly-skilled technology journalist.
Domain: {cat_label}
Topic/Headline Focus: {topic}

Extracted Live Search Context:
{news_context}

{validator_feedback}

Your task is to synthesize the extracted context above into a cohesive, highly engaging technical news roundup blog post in Markdown format.
Focus on factual accuracy, properly summarizing the facts from the search context.
Use a professional journalism tone, appropriate headers, and bold critical terms. Do not include a markdown codeblock around your entire response.
"""
