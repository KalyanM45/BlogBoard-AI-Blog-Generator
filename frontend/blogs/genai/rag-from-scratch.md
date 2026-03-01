# Building RAG Pipelines from Scratch

Retrieval-Augmented Generation (RAG) is one of the most practical techniques in modern AI engineering. It grounds LLM outputs in real, verifiable data — dramatically reducing hallucinations and enabling knowledge-base Q&A without expensive fine-tuning.

In this article, we'll build a complete RAG pipeline from scratch using Python, `sentence-transformers`, and `FAISS`.

---

## Why RAG?

LLMs are powerful but suffer from:

- **Knowledge cutoffs** — they don't know recent events
- **Hallucination** — they confidently make things up
- **Privacy** — you can't fine-tune on proprietary data

RAG solves this by retrieving relevant context at inference time:

```
Query → Retrieve relevant docs → Augment prompt → Generate grounded answer
```

---

## Architecture Overview

```
┌──────────────────────────────────────────────┐
│                 RAG Pipeline                 │
│                                              │
│  Documents → Chunking → Embedding → FAISS    │
│                                    ↓         │
│  Query → Embed → Search → Top-K chunks       │
│                              ↓               │
│          [System Prompt + Chunks + Query]     │
│                              ↓               │
│                           LLM → Answer       │
└──────────────────────────────────────────────┘
```

---

## Step 1: Document Chunking

Chunking strategy dramatically affects retrieval quality:

```python
def chunk_text(text, chunk_size=512, overlap=64):
    """
    Split text into overlapping chunks.
    Overlap preserves context across chunk boundaries.
    """
    words = text.split()
    chunks = []
    start = 0
    while start < len(words):
        end = min(start + chunk_size, len(words))
        chunk = ' '.join(words[start:end])
        chunks.append(chunk)
        start += chunk_size - overlap
    return chunks
```

**Chunking strategies:**
- **Fixed-size** — Simple but may cut mid-sentence
- **Sentence-based** — Better semantic coherence
- **Paragraph-based** — Natural structure
- **Semantic splitting** — Use embeddings to find natural breaks

---

## Step 2: Embedding

Convert chunks into vector representations:

```python
from sentence_transformers import SentenceTransformer

model = SentenceTransformer('BAAI/bge-large-en-v1.5')  # Strong retrieval model

def embed_chunks(chunks):
    embeddings = model.encode(
        chunks,
        normalize_embeddings=True,  # Cosine similarity
        show_progress_bar=True,
        batch_size=32
    )
    return embeddings  # shape: (N, 1024)
```

**Top embedding models for RAG:**

| Model | Dims | Notes |
|---|---|---|
| `BAAI/bge-large-en-v1.5` | 1024 | Best quality, MTEB leader |
| `text-embedding-3-small` | 1536 | OpenAI, great quality |
| `nomic-embed-text-v1.5` | 768 | Open source, fast |

---

## Step 3: Vector Store with FAISS

```python
import faiss
import numpy as np
import pickle

class VectorStore:
    def __init__(self, dim):
        self.index = faiss.IndexFlatIP(dim)  # Inner product = cosine if normalized
        self.chunks = []

    def add(self, chunks, embeddings):
        self.chunks.extend(chunks)
        self.index.add(np.array(embeddings, dtype='float32'))

    def search(self, query_embedding, k=5):
        q = np.array([query_embedding], dtype='float32')
        scores, indices = self.index.search(q, k)
        return [(self.chunks[i], float(scores[0][j]))
                for j, i in enumerate(indices[0]) if i >= 0]

    def save(self, path):
        faiss.write_index(self.index, f"{path}.faiss")
        with open(f"{path}.pkl", 'wb') as f:
            pickle.dump(self.chunks, f)

    def load(self, path):
        self.index = faiss.read_index(f"{path}.faiss")
        with open(f"{path}.pkl", 'rb') as f:
            self.chunks = pickle.load(f)
```

---

## Step 4: Retrieval

```python
def retrieve(query, store, embed_model, k=5):
    query_emb = embed_model.encode(query, normalize_embeddings=True)
    results = store.search(query_emb, k=k)
    return results  # [(chunk_text, score), ...]
```

---

## Step 5: Augmented Generation

```python
from openai import OpenAI

client = OpenAI()

def rag_query(query, store, embed_model, k=5):
    # 1. Retrieve
    results = retrieve(query, store, embed_model, k)
    context = "\n\n---\n\n".join([chunk for chunk, _ in results])

    # 2. Build prompt
    system_prompt = """You are a helpful assistant. Answer the user's question 
using ONLY the provided context. If the answer cannot be found in the context, 
say "I don't have enough information to answer this."

Context:
{context}""".format(context=context)

    # 3. Generate
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": query}
        ],
        temperature=0.1  # Low temp for factual answers
    )

    return response.choices[0].message.content
```

---

## Evaluation: How Do You Know It's Working?

Use **RAGAS** metrics:

- **Faithfulness** — Does the answer stay grounded in the retrieved context?
- **Answer Relevancy** — Does the answer address the question?
- **Context Precision** — Are the retrieved chunks actually relevant?
- **Context Recall** — Are all necessary chunks retrieved?

```python
from ragas import evaluate
from ragas.metrics import faithfulness, answer_relevancy

# ... run evaluation
```

---

## Advanced Techniques

### Hybrid Search
Combine semantic (vector) search with keyword (BM25) search:

```python
from rank_bm25 import BM25Okapi

# BM25 for keyword match + FAISS for semantic → merge scores
```

### Reranking
After top-K retrieval, use a cross-encoder to re-rank:

```python
from sentence_transformers import CrossEncoder

reranker = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')
scores = reranker.predict([[query, chunk] for chunk, _ in results])
```

### HyDE (Hypothetical Document Embeddings)
Generate a hypothetical answer first, embed it, then retrieve:

```python
# 1. Ask LLM to generate a hypothetical answer
# 2. Embed that answer
# 3. Use it as the retrieval query
```

---

## Full Pipeline Example

```python
# Build index (run once)
documents = load_documents('./docs/')
chunks = []
for doc in documents:
    chunks.extend(chunk_text(doc))

embed_model = SentenceTransformer('BAAI/bge-large-en-v1.5')
embeddings = embed_chunks(chunks)

store = VectorStore(dim=1024)
store.add(chunks, embeddings)
store.save('my_kb')

# Query (at inference)
store.load('my_kb')
answer = rag_query("What is the return policy?", store, embed_model)
print(answer)
```

---

*RAG is currently the most production-proven way to ground LLMs in domain-specific knowledge. Master it, and you have a powerful tool for almost any enterprise AI use case.*
