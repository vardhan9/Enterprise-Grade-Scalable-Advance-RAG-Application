# Enterprise-Grade Advanced RAG Application

A production-oriented RAG engineering project focused on **retrieval quality, reliability, observability, guardrails, and measurable improvement**.

> **Core idea:** Don't just build a RAG system. Build one you can measure, debug, and improve.

---

## 1. What This Project Demonstrates

![alt text](image-1.png)

![alt text](image.png)

![alt text](image-2.png)

---

## 2. Main Components

| Layer | Technology | Purpose |
|---|---|---|
| UI | Streamlit | Chat + evaluation |
| API | FastAPI | Application backend |
| Orchestration | LangGraph | Routing + state |
| Vector DB | Qdrant | Semantic retrieval |
| Embeddings | Sentence Transformers | Query/document vectors |
| Reranker | FlashRank | Relevance refinement |
| Guardrails | NeMo + deterministic checks | Safety/control |
| LLM Gateway | Portkey | Provider abstraction |
| Observability | Pydantic Logfire | Tracing + monitoring |
| Evaluation | RAGAS + custom metrics | Quality measurement |

---

## 3. End-to-End RAG Flow

```text
User Query
    ↓
Guardrails
    ↓
LangGraph Planner
    ↓
Query Embedding
    ↓
Qdrant Top-N Retrieval
    ↓
FlashRank Reranking
    ↓
Top-K Context
    ↓
LLM Generation
    ↓
Answer + Sources
```

Conversational requests can bypass document retrieval when appropriate.

---

## 4. Document Ingestion

Supported formats:

- PDF
- HTML
- TXT
- DOCX
- PPTX

```text
Source
  ↓
Parser
  ↓
Text Extraction
  ↓
Chunking
  ↓
Embeddings
  ↓
Qdrant
```

Current retrieval embedding:

`sentence-transformers/all-mpnet-base-v2`

Vector dimension: **768**

---

## 5. Retrieval + Reranking

The system uses a two-stage retrieval strategy:

```text
Query
  ↓
Qdrant
  ↓
Top 15 candidates
  ↓
FlashRank
  ↓
Top 5 relevant chunks
```

Why?

- Vector search → fast candidate discovery
- Cross-encoder → better relevance ordering

If reranking fails, the system can fall back to the original retrieval results.

---

## 6. Agentic Orchestration

LangGraph separates routing from execution:

```text
                 ┌─ Conversational → Responder
Planner ─────────┤
                 └─ Technical → Retriever → Responder
```

Conversation state is maintained with a `thread_id`.

This keeps routing, retrieval, and generation as separate concerns.

---

## 7. Guardrails

Two control layers:

```text
User Input
    ↓
Deterministic Checks
    ↓
NeMo Guardrails
    ↓
Allow / Block
```

A blocked request exits before the RAG pipeline.

Important production lesson:

> **A successful LLM call does not necessarily mean a successful application decision.**

Machine-level guardrail decisions should be validated and unexpected outputs should fail closed.

---

## 8. LLM Gateway

The application uses Portkey as an OpenAI-compatible gateway:

```text
Application
    ↓
OpenAI-compatible Client
    ↓
Portkey
    ↓
Saved Provider / Integration
    ↓
Model
```

This provides a separation between application logic and model providers.

Gateway capabilities can include:

- Routing
- Retries
- Fallbacks
- Caching
- Metadata
- Centralized controls

---

## 9. Observability

Logfire traces important pipeline stages:

- Guardrails
- Planner
- Retrieval
- Reranking
- LLM generation
- Ingestion
- Embeddings
- Evaluation

The goal is to answer:

> **What happened, where, and why?**

---

## 10. Evaluation

The project uses a repeatable golden dataset and evaluates the live pipeline.

### RAG metrics

| Metric | Measures |
|---|---|
| Faithfulness | Is the answer supported by context? |
| Answer Relevancy | Does the answer address the question? |
| Context Precision | Are relevant chunks ranked higher? |
| Context Recall | Did retrieval find the required evidence? |
| Answer Correctness | How close is the answer to the reference? |
| Tool Correctness | Was the expected route/tool selected? |

### Guardrail metrics

`Precision · Recall · Accuracy`

---

## 11. Baseline: Failure as a Diagnostic Signal

One observed baseline:

```text
Faithfulness        0.089
Answer Relevancy    0.885
Context Precision   0.214
Context Recall     0.095
Answer Correctness  0.607
Tool Correctness   1.000
```

The important observation is not the individual number.

It is the pattern:

```text
High answer relevance
        +
Low retrieval quality
        +
Low faithfulness
```

This points investigation toward:

**Parsing → Chunking → Embeddings → Retrieval → Reranking → Context**

before simply changing the LLM.

> These are experimental baseline results, not production guarantees.

---

## 12. Evaluation Workflow

```text
Golden Dataset
      ↓
Run Live Pipeline
      ↓
Capture Responses + Context
      ↓
RAGAS / Guardrail Metrics
      ↓
Analyze
      ↓
Change One Variable
      ↓
Re-run
```

This turns:

> “This version feels better.”

into:

> “This version changed measurable system behavior.”

---

## 13. Experimentation

Experiment dimensions include:

- Chunk size / strategy
- Embedding model
- Retrieval top-K
- Reranking
- Prompt
- LLM
- Context limits
- Routing
- Guardrail strategy

Use the **same golden dataset** when comparing changes.

---

## 14. Performance & Reliability

The system includes several operational controls:

- Batch document embedding
- Bounded generation context
- Lazy model initialization
- Retrieval fallback
- Evaluation throttling
- Independent tracing
- Provider/gateway abstraction

The goal is not only speed.

It is **predictable system behavior**.

---

## 15. Project Structure

```text
app/
├── ingestion/
├── services/
│   └── retrieval/
├── agents/
├── guardrails/
├── gateway/
├── main.py
└── config.py

evals/
├── golden_dataset.json
├── pipeline.py
├── metrics.py
├── guardrails_eval.py
└── app.py

ui/
└── app.py

DATA/
├── true_data/
└── noisy_data/

DOCS/
```

---

## 16. Run Locally

### Install

```bash
pip install -r requirements.txt
```

### Environment

Configure the required service credentials in `.env`.

Typical services:

```text
QDRANT_API_KEY
QDRANT_CLUSTER_ENDPOINT
PORTKEY_API_KEY
GROQ_API_KEY
LOGFIRE_TOKEN
```

Never commit real credentials.

### Ingest

```bash
python -m app.ingestion.processor DATA --wipe
```

### Start API

```bash
uvicorn app.main:app --reload --port 8000
```

### Start Chat UI

```bash
streamlit run ui/app.py
```

### Start Evaluation UI

```bash
streamlit run evals/app.py
```

---

## 17. Deployment

The application is container-friendly and can be deployed using Docker and cloud infrastructure.

Production considerations include:

- Environment configuration
- Monitoring
- Logging
- Rate limits
- Cost controls
- Load testing
- CI/CD evaluation gates

---

## 18. Continuous Improvement

```text
BUILD
  ↓
OBSERVE
  ↓
EVALUATE
  ↓
DIAGNOSE
  ↓
OPTIMIZE
  ↓
REPEAT
```

### Core lesson

**RAG quality is a system property.**

A stronger LLM cannot reliably compensate for poor retrieval.

A strong retriever cannot compensate for ungrounded generation.

The engineering task is to make every layer:

**measurable · explainable · reliable · improvable**

---

## 19. Limitations / Next Steps

Planned areas include:

- Hybrid dense + sparse retrieval
- Better structure-aware chunking
- Metadata filtering
- Dynamic top-K
- Reranker benchmarking
- Larger evaluation datasets
- Human evaluation calibration
- P50/P95/P99 latency
- Cost-per-query tracking
- Stronger structured guardrail outputs
- Automated regression gates

---

## 20. Project Summary

**Architecture:** Agentic RAG + LangGraph  
**API:** FastAPI  
**UI:** Streamlit  
**Vector DB:** Qdrant  
**Embeddings:** Sentence Transformers  
**Reranker:** FlashRank  
**Guardrails:** NeMo + deterministic checks  
**LLM Gateway:** Portkey  
**Observability:** Pydantic Logfire  
**Evaluation:** RAGAS + custom metrics  

### Build → Measure → Learn → Improve

This project is an engineering environment for **building, breaking, measuring, and improving RAG systems**.
