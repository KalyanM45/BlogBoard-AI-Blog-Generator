# Prompt Engineering: Advanced Patterns and Techniques

Prompt engineering is not about tricks — it's about understanding how LLMs process information and structuring inputs to elicit the best possible outputs. This guide covers the essential advanced patterns every AI practitioner should know.

---

## Why Prompting Matters

LLMs are fundamentally **instruction-following, context-sensitive** systems. The same question, framed differently, can produce wildly different quality answers. Effective prompting:

- Reduces hallucinations
- Increases consistency and format compliance
- Improves multi-step reasoning
- Enables complex task decomposition

---

## Pattern 1: Chain-of-Thought (CoT)

Standard prompting asks directly; CoT prompts the model to reason step-by-step before answering:

**Without CoT:**
```
Q: Roger has 5 tennis balls. He buys 2 more cans of 3 balls. How many does he have?
A: 11
```

**With CoT:**
```
Q: Roger has 5 tennis balls. He buys 2 more cans of 3 balls. How many does he have?
   Let's think step by step.

A: Roger starts with 5 balls.
   2 cans × 3 balls = 6 new balls.
   5 + 6 = 11 balls total.
   Answer: 11
```

The magic phrase: **"Let's think step by step"** or **"Think through this carefully before answering."**

```python
def cot_prompt(question):
    return f"""Solve the following problem step by step, showing your reasoning clearly.

Problem: {question}

Step-by-step reasoning:"""
```

---

## Pattern 2: Few-Shot Prompting

Provide examples to teach the model the output format and task structure:

```python
few_shot_prompt = """Classify the sentiment of each review as POSITIVE, NEGATIVE, or NEUTRAL.

Review: "The product quality is exceptional, well worth the price."
Sentiment: POSITIVE

Review: "It works fine but nothing special about it."
Sentiment: NEUTRAL

Review: "Broke after two days, completely useless."
Sentiment: NEGATIVE

Review: "{user_review}"
Sentiment:"""
```

**Tips for few-shot examples:**
- Use 3-8 examples (diminishing returns after that)
- Cover edge cases and ambiguous cases
- Keep format consistent
- Order matters — the last example has the most influence

---

## Pattern 3: System Prompt Engineering

The system prompt defines the model's persona, constraints, and behavior:

```python
system_prompt = """You are an expert ML engineer with 10+ years of experience.

Your communication style:
- Be precise and technical, but also clear
- Always provide working code examples in Python
- Acknowledge uncertainty honestly — say "I'm not sure" when appropriate
- Structure responses with headers for longer explanations

Constraints:
- Only discuss topics related to machine learning, data science, and AI
- Do not generate harmful, unethical, or misleading content
- Always recommend testing and validation for any code you provide"""
```

---

## Pattern 4: Self-Consistency

Generate multiple reasoning chains, then take the majority vote:

```python
from openai import OpenAI
from collections import Counter

client = OpenAI()

def self_consistent_answer(question, n=5):
    """Generate multiple reasoning paths and vote."""
    answers = []
    for _ in range(n):
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "user", "content": f"{question}\n\nThink step by step."}
            ],
            temperature=0.7  # Non-zero temp for diversity
        )
        # Extract the final answer (last sentence or specific format)
        answer = extract_final_answer(response.choices[0].message.content)
        answers.append(answer)
    
    # Majority vote
    return Counter(answers).most_common(1)[0][0]
```

---

## Pattern 5: Structured Output

Force the model to output valid JSON for downstream processing:

```python
structured_prompt = """Extract the key information from the following job description.

Return ONLY a JSON object with this exact structure:
{
  "title": "job title",
  "company": "company name",
  "required_skills": ["skill1", "skill2"],
  "experience_years": <number or null>,
  "location": "location or 'Remote'",
  "salary_range": "range or null"
}

Job description:
{job_description}

JSON output:"""
```

Or use the newer `response_format` parameter:

```python
from pydantic import BaseModel
from openai import OpenAI

class JobExtraction(BaseModel):
    title: str
    company: str
    required_skills: list[str]
    experience_years: int | None
    location: str

client = OpenAI()
response = client.beta.chat.completions.parse(
    model="gpt-4o",
    messages=[{"role": "user", "content": job_description}],
    response_format=JobExtraction
)
result = response.choices[0].message.parsed
```

---

## Pattern 6: ReAct (Reason + Act)

For agentic tasks, combine reasoning with actions:

```
Thought: I need to find the current Bitcoin price.
Action: search("Bitcoin current price USD")
Observation: Bitcoin is currently $65,432.

Thought: Now I need to calculate 10% of this.
Action: calculate("65432 * 0.10")
Observation: 6543.2

Thought: I have all the information I need.
Final Answer: 10% of the current Bitcoin price ($65,432) is $6,543.20.
```

---

## Anti-Patterns to Avoid

| Anti-Pattern | Problem | Fix |
|---|---|---|
| Vague instructions | Inconsistent output | Be specific about format and scope |
| Huge context dumps | Loss of focus | Chunk and summarize first |
| Negative framing | Models are better at "do X" than "don't do Y" | Rephrase positively |
| No examples | High variance outputs | Add 2-3 examples |
| Asking multiple questions at once | Partial answers | Break into sequential prompts |

---

## Evaluation Framework

```python
def evaluate_prompt(prompt_template, test_cases, expected_outputs):
    scores = []
    for case, expected in zip(test_cases, expected_outputs):
        prompt = prompt_template.format(**case)
        response = get_model_response(prompt)
        score = compute_similarity(response, expected)
        scores.append(score)
    return sum(scores) / len(scores)
```

Always evaluate prompts systematically — what works for one use case may fail for another.

---

*Great prompting is a learnable skill. The engineers who master it are building the most effective AI systems in the world today.*
