TUTORIAL_TOPIC_PROMPT = """
You are an expert content strategist for a technical blog.
Domain: {cat_label}

Recent History of articles published in this domain:
{history}

Your task is to select an exciting, novel topic that hasn't been covered in the recent history above.
Return the result strictly as JSON:
{{
  "topic": "The title of the new topic",
  "subtopics": "A comma-separated list of 3-4 subtopics to cover"
}}
"""

TUTORIAL_GENERATION_PROMPT = """
You are a highly skilled technical writer.
Domain/Category: {cat_label}
Topic: {topic}
Subtopics to cover: {subtopics}

{validator_feedback}

Your task is to write a comprehensive, highly engaging, and in-depth tutorial blog post in Markdown format.
Use a professional tone, appropriate headers, and bold critical terms. Do not include a markdown codeblock around your entire response.
"""
