# Word Embeddings: From Word2Vec to Contextual Representations

Understanding word embeddings is foundational to NLP. They transform symbolic text into geometric space where meaning becomes distance — and allow models to reason about language algebraically.

---

## The Limitations of One-Hot Encoding

Before embeddings, words were represented as sparse vectors:

```
Vocabulary: [cat, dog, fish, bird]
"cat" = [1, 0, 0, 0]
"dog" = [0, 1, 0, 0]
```

**Problems:**
- Dimensionality explodes with vocabulary size (often 50k-100k tokens)
- No semantic similarity — `cos("cat", "dog") = 0`
- No generalization — model must see each word independently

---

## Word2Vec: Learning Meaning from Context

Word2Vec (Mikolov et al., 2013) learns dense embeddings from word co-occurrence. Two architectures:

### CBOW (Continuous Bag of Words)
Predict center word from surrounding context:

```
Context: ["The", _, "sat", "on"] → Target: "cat"
```

### Skip-gram
Predict surrounding context from center word:

```
Center: "cat" → Predict: ["The", "sat", "on", "mat"]
```

```python
from gensim.models import Word2Vec

sentences = [["the", "cat", "sat", "on", "mat"],
             ["the", "dog", "ran", "in", "park"]]

model = Word2Vec(sentences, vector_size=100, window=5, min_count=1, sg=1)

# Semantic similarity
print(model.wv.similarity("cat", "dog"))  # ~0.78

# Famous analogy: king - man + woman = queen
result = model.wv.most_similar(
    positive=["king", "woman"], negative=["man"], topn=1
)
print(result)  # [('queen', 0.91)]
```

---

## GloVe: Global Co-occurrence Statistics

GloVe (Pennington et al., 2014) combines local context (Word2Vec) with global statistics:

$$J = \sum_{i,j} f(X_{ij})\left(\mathbf{w}_i^T \tilde{\mathbf{w}}_j + b_i + \tilde{b}_j - \log X_{ij}\right)^2$$

Where `X_{ij}` is the co-occurrence count of words `i` and `j`.

---

## The Problem: Polysemy

Static embeddings assign **one vector per word**, regardless of context:

> "I deposited money at the **bank**."  
> "I sat by the river **bank**."

Both uses of "bank" would have the same embedding — problematic!

---

## ELMo: Contextualized Embeddings

ELMo (Embeddings from Language Models, 2018) generates embeddings based on the full sentence context using a pre-trained bidirectional LSTM:

```python
import tensorflow_hub as hub

elmo = hub.Module("https://tfhub.dev/google/elmo/3")

# Now "bank" has different vectors in different contexts
embeddings = elmo(["I sat by the river bank", "I used the bank ATM"])
```

---

## BERT: Transformers Change Everything

BERT (Bidirectional Encoder Representations from Transformers) takes contextualization much further:

```python
from transformers import BertTokenizer, BertModel
import torch

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertModel.from_pretrained('bert-base-uncased')

sentence = "The bank by the river is beautiful"
inputs = tokenizer(sentence, return_tensors="pt")
with torch.no_grad():
    outputs = model(**inputs)

# Last hidden state: (batch, seq_len, 768)
embeddings = outputs.last_hidden_state
print(embeddings.shape)  # torch.Size([1, 9, 768])
```

---

## Comparison Table

| Method | Year | Context | Dimensionality | Polysemy |
|---|---|---|---|---|
| One-Hot | - | ❌ | Vocab size | ❌ |
| Word2Vec | 2013 | Local window | 100-300 | ❌ |
| GloVe | 2014 | Global stats | 100-300 | ❌ |
| ELMo | 2018 | Sentence | 1024 | ✅ |
| BERT | 2018 | Full doc | 768 | ✅✅ |

---

## Practical Tips

- Use **GloVe/Word2Vec** for lightweight similarity tasks and small models
- Use **BERT/sentence-transformers** for semantic search and classification
- For retrieval, prefer models fine-tuned for cosine similarity like `BAAI/bge-*`
- Dimensionality ≠ quality — a well-trained 128-dim model can outperform a poor 1024-dim model

---

*Word embeddings are the bridge between human language and mathematical reasoning. Understanding them deeply will make you a better NLP practitioner.*
