ENTERPRISE-GRADE SCALABLE ADVANCED RAG APPLICATION
====================================================

A production-oriented Retrieval-Augmented Generation (RAG) reference implementation focused on retrieval quality, agentic orchestration, reranking, guardrails, LLM gateway integration, observability, and measurable evaluation.

Repository branch:
    local_dev

Project status:
    Experimental / engineering benchmark
    The project is intentionally designed to expose failure modes and make RAG improvements measurable rather than relying on subjective "looks good" answers.


1. PROJECT AIM
==============

The goal of this project is not simply to build a chatbot.

The goal is to build and evaluate an end-to-end RAG system as an engineering system:

    Documents
        |
        v
    Parsing
        |
        v
    Chunking
        |
        v
    Embeddings
        |
        v
    Qdrant Vector Search
        |
        v
    Cross-Encoder Reranking
        |
        v
    Context Construction
        |
        v
    LLM Generation
        |
        v
    Answer

with additional control layers:

    Guardrails + Agentic Routing + LLM Gateway + Observability + Evaluation

The central engineering question is:

    "How do we know that a RAG system is actually getting better?"

The answer implemented here is:

    Measure it.

The project therefore treats retrieval, generation, safety, tool usage, latency, and operational behavior as separate engineering concerns.


2. WHY THIS PROJECT EXISTS
==========================

A RAG demo can produce impressive answers while hiding serious problems.

A system may:

- retrieve the wrong chunks,
- retrieve only part of the required evidence,
- rank irrelevant chunks too highly,
- generate an answer that sounds correct but is not grounded,
- call an unnecessary tool,
- fail to handle unsafe or off-topic requests,
- waste LLM tokens,
- become difficult to debug when something goes wrong.

This project was built to make those failure modes visible.

Instead of evaluating the system with a handful of manually tested questions, the repository contains a golden evaluation dataset, a repeatable live-pipeline evaluation phase, RAGAS-based scoring, guardrail classification metrics, and a Streamlit evaluation dashboard.


3. CORE DESIGN PHILOSOPHY
=========================

The architecture follows several principles:

1. Separate retrieval quality from generation quality.
2. Treat reranking as an independent retrieval stage.
3. Keep safety decisions outside the generation prompt where possible.
4. Make LLM calls observable.
5. Use a gateway layer rather than coupling application code directly to a provider.
6. Use golden test cases as regression tests.
7. Diagnose failures using metrics instead of intuition.
8. Optimize the system, not only the model.


4. HIGH-LEVEL ARCHITECTURE
==========================

                                +----------------------+
                                |   Streamlit UI       |
                                |  Chat + Eval Suite   |
                                +----------+-----------+
                                           |
                                           v
                                +----------------------+
                                |      FastAPI API     |
                                +----------+-----------+
                                           |
                                  Guardrails Gate
                                           |
                              +------------+------------+
                              |                         |
                           BLOCK                     ALLOW
                              |                         |
                            STOP                 LangGraph Agent
                                                        |
                                               +--------+--------+
                                               |                 |
                                         Conversational     Technical
                                               |                 |
                                               |             Retrieval
                                               |                 |
                                               |          +------+------+
                                               |          |             |
                                               |       Qdrant       FlashRank
                                               |       Search       Reranker
                                               |          |             |
                                               |          +------+------+
                                               |                 |
                                               +--------+--------+
                                                        |
                                                   LLM Response
                                                        |
                                                        v
                                                   Final Answer


Cross-cutting infrastructure:

    Portkey       -> LLM gateway / provider routing / gateway-level controls
    Logfire       -> tracing and operational observability
    Qdrant        -> vector retrieval
    RAGAS         -> RAG quality evaluation
    Golden Set    -> repeatable regression benchmark


5. END-TO-END RAG FLOW
=====================

A technical request follows this path:

    User Query
       |
       v
    NeMo Guardrails
       |
       v
    LangGraph Planner
       |
       v
    Search Query
       |
       v
    Local Embedding Model
       |
       v
    Qdrant
       |
       v
    Top-N Candidate Retrieval
       |
       v
    FlashRank Cross-Encoder
       |
       v
    Top-5 Context
       |
       v
    Context Construction
       |
       v
    LLM Synthesis
       |
       v
    Response + Retrieved Sources


For conversational queries, the planner can route directly to response generation and skip document retrieval.

The LangGraph checkpointer provides conversation memory through a thread_id.


6. DOCUMENT INGESTION
=====================

The ingestion engine supports multiple document formats.

Current parsers include:

    PDF      -> pypdf with pdfplumber fallback
    HTML     -> BeautifulSoup
    TXT      -> local text parser
    DOCX     -> Unstructured
    PPTX     -> Unstructured

The ingestion pipeline is:

    Source File
       |
       v
    File-Type Parser
       |
       v
    Text Extraction
       |
       v
    Chunking
       |
       v
    Local Processed JSON
       |
       v
    Embedding
       |
       v
    Qdrant Upsert


Processed chunk metadata is stored locally under:

    processed_data/


7. CHUNKING
===========

The current chunker is intentionally simple and deterministic.

It:

- splits extracted text around paragraph boundaries,
- accumulates paragraphs until the configured character limit,
- emits non-empty chunks,
- records the process through Logfire.

The current default is approximately:

    chunk_size = 1500 characters

This is deliberately an experiment-friendly design.

A future optimization area is to compare:

    paragraph-based chunking
    fixed token chunking
    recursive chunking
    structure-aware chunking
    semantic chunking

against the same golden dataset.


8. EMBEDDINGS
=============

The local retrieval path uses:

    sentence-transformers/all-mpnet-base-v2

Vector dimension:

    768

Qdrant is configured with:

    distance = COSINE

Embedding models are loaded lazily and reused within the process.

The same local embedding implementation is used for query embedding during retrieval and can therefore keep the query/document embedding space consistent.


9. VECTOR DATABASE — QDRANT
===========================

Qdrant stores:

    vector
    text
    source
    source_type

The retrieval service uses Qdrant's modern query_points interface.

Current retrieval strategy:

    Query
      |
      v
    Embed query
      |
      v
    Qdrant semantic search
      |
      v
    15 candidate chunks


The top candidates are then passed to the reranking stage.


10. SEMANTIC RERANKING
======================

Vector similarity is fast and scalable, but initial retrieval is still approximate.

This project therefore adds a second retrieval stage:

    Qdrant
       |
       | top 15
       v
    FlashRank
       |
       | cross-encoder reranking
       v
    top 5


FlashRank runs locally using an optimized ONNX-based ranking model.

The design also contains a fallback:

    If reranking fails
        |
        v
    return the original top documents


This protects the application from turning a reranker failure into a complete retrieval failure.


11. AGENTIC ORCHESTRATION
=========================

LangGraph manages the request flow.

The current graph contains three main nodes:

    Planner
       |
       +----> Conversational ----> Responder
       |
       +----> Technical ---------> Retriever ----> Responder


The planner decides whether the latest message can be answered from conversation context or whether documentation retrieval is required.

The graph uses a MemorySaver checkpointer so conversations can be associated with a thread_id.


12. RESPONSE GENERATION
=======================

The responder combines:

    - current user question
    - conversation history
    - retrieved technical context

For technical questions, the system instructs the model to answer using the supplied technical context.

For conversational questions, document retrieval is skipped and conversation memory is used.

The implementation also protects the generation request from excessive context by applying a context character limit before synthesis.


13. GUARDRAILS
=============

NeMo Guardrails is used as an input control layer.

The guardrail architecture has two layers:

    Layer 1
    Deterministic checks
        |
        v
    Layer 2
    NeMo Guardrails / LLM-based rails
        |
        v
    Allow or block


The deterministic layer includes configured jailbreak and off-topic patterns.

The NeMo layer provides additional conversational and safety control through Colang configuration.

A blocked request exits before the LangGraph RAG pipeline and returns:

    answer
    status
    guardrail trace
    empty sources


An important engineering lesson from this project is that:

    "LLM call succeeded"

does not necessarily mean:

    "application decision succeeded."

Production guardrails should validate machine-level decisions and fail closed on unexpected classifier outputs rather than allowing arbitrary model text to become an implicit control signal.


14. LLM GATEWAY — PORTKEY
========================

The application uses Portkey as an OpenAI-compatible LLM gateway.

The application-facing pattern is:

    Application
        |
        v
    OpenAI-compatible client
        |
        v
    Portkey
        |
        v
    Saved provider/integration
        |
        v
    Model provider


This keeps the application layer decoupled from the underlying model provider.

The gateway integration is also designed to support operational capabilities such as:

    - provider routing
    - gateway-level metadata
    - retries
    - fallback strategies
    - caching
    - centralized LLM observability

The repository contains gateway configuration examples for fallback, retry, and caching. The exact active behavior depends on the Portkey saved integration/configuration used by the deployment.


15. OBSERVABILITY
================

Pydantic Logfire is used throughout the application and evaluation stack.

Important spans include:

    - user interaction
    - guardrail checks
    - planner decisions
    - Qdrant retrieval
    - semantic reranking
    - LLM synthesis
    - ingestion
    - embedding batches
    - evaluation phases


This creates a traceable execution path rather than treating the LLM as a black box.

A useful production distinction is:

    Model-level observability
        What did the model generate?

    System-level observability
        What decision did the application make with that output?


The second is critical for agentic systems.


16. EVALUATION STRATEGY
======================

The project contains an evaluation suite rather than relying on manual inspection.

The evaluation pipeline has three major stages:

    Phase 1
    Golden Dataset
        |
        v
    Phase 2
    Execute the live RAG pipeline
        |
        v
    Phase 3
    Score the captured results


The golden dataset currently contains:

    15 RAG samples
    6 guardrail test cases


The RAG samples are derived from five enterprise-style source documents in DATA/true_data/.


17. GOLDEN DATASET
==================

Each RAG golden contains information such as:

    id
    domain
    question
    reference answer
    relevant contexts
    expected tools
    actual response
    actual contexts
    actual tools called


The important idea is that the dataset becomes a stable benchmark.

Whenever you change:

    chunking
    embedding model
    retrieval strategy
    top-K
    reranker
    prompt
    LLM
    gateway configuration

you can run the same benchmark again.

This turns RAG development from:

    "This version feels better."

into:

    "This version changed measurable system behavior."


18. RAGAS EVALUATION
===================

The evaluation suite measures six dimensions:

    1. Faithfulness
    2. Answer Relevancy
    3. Context Precision
    4. Context Recall
    5. Answer Correctness
    6. Tool Correctness


Metric interpretation:

    Faithfulness
        Is the generated answer supported by retrieved context?

    Answer Relevancy
        Does the answer address the user's question?

    Context Precision
        Are relevant retrieved chunks ranked above irrelevant ones?

    Context Recall
        Did retrieval capture the information required to answer the question?

    Answer Correctness
        How closely does the generated answer match the trusted reference?

    Tool Correctness
        Did the agent select/use the expected tool or route?


This separation is important.

A system can have high answer relevance while having poor faithfulness.

It can also retrieve useful evidence but fail during answer generation.

Therefore:

    Retrieval quality != Generation quality


19. CURRENT BASELINE RESULTS
============================

One observed baseline evaluation run produced:

    Faithfulness        0.089   Poor
    Answer Relevancy    0.885   Good
    Context Precision   0.214   Poor
    Context Recall     0.095   Poor
    Answer Correctness  0.607   Fair
    Tool Correctness   1.000   Good


These numbers are not presented as a final production benchmark.

They are valuable because they reveal a diagnostic pattern:

    High Answer Relevancy
            +
    Very low Context Recall
            +
    Very low Context Precision
            +
    Very low Faithfulness

This suggests that the system can generate an answer that is relevant to the question while failing to retrieve enough high-quality evidence to ground that answer.

That points the next engineering investigation toward:

    parsing
    chunking
    embeddings
    retrieval
    top-K
    reranking
    context construction

rather than immediately blaming the generation model.


20. GUARDRAILS EVALUATION
========================

Guardrail tests are evaluated as a binary classification problem.

Each test becomes one of:

    TP  True Positive
    TN  True Negative
    FP  False Positive
    FN  False Negative


From these the system calculates:

    Precision
    Recall
    Accuracy


This makes safety behavior measurable rather than subjective.

For example:

    Precision
        Of the requests the guardrail blocked,
        how many should actually have been blocked?

    Recall
        Of all requests that should have been blocked,
        how many did the guardrail catch?


21. EVALUATION DASHBOARD
=======================

The Streamlit evaluation application provides three major stages:

    Step 1
    Review Ground Truth

    Step 2
    Run Live Pipeline

    Step 3
    Run Evaluation Metrics


The dashboard exposes:

    - golden questions
    - reference answers
    - expected tools
    - live responses
    - retrieved contexts
    - guardrail TP/TN/FP/FN results
    - RAGAS metric scores
    - per-sample evaluation results
    - aggregate scores


This turns the evaluation process into a repeatable engineering workflow.


22. PERFORMANCE AND OPERATIONAL CONSIDERATIONS
==============================================

The repository includes several production-oriented considerations.

Lazy initialization:
    Heavy models such as the reranker are loaded only when first needed.

Batch embedding:
    Documents are embedded in batches rather than one request at a time.

Context limits:
    Generation context is bounded to avoid unnecessarily large LLM requests.

Evaluation throttling:
    Evaluation calls include batching and cooldowns to respect provider rate/token limits.

Fallback behavior:
    Retrieval can fall back to the original vector-search ordering if reranking fails.

Observability:
    Important pipeline stages are traced independently.

These are not merely performance tricks. They are reliability controls.


23. PROJECT STRUCTURE
=====================

    app/
    |
    +-- ingestion/
    |   +-- loaders/
    |   |   +-- pdf.py
    |   |   +-- html.py
    |   |   +-- office.py
    |   |   +-- text.py
    |   |
    |   +-- chunking/
    |       +-- splitter.py
    |
    +-- services/
    |   +-- retrieval/
    |       +-- embeddings.py
    |       +-- embeddings_local.py
    |       +-- qdrant_service.py
    |       +-- ranking_service.py
    |
    +-- agents/
    |   +-- state.py
    |   +-- graph.py
    |   +-- nodes/
    |       +-- planner.py
    |       +-- retriever.py
    |       +-- responder.py
    |
    +-- guardrails/
    |   +-- rails.py
    |   +-- colang_rules.py
    |
    +-- gateway/
    |   +-- client.py
    |
    +-- main.py
    +-- config.py

    evals/
    |
    +-- golden_dataset.json
    +-- pipeline.py
    +-- metrics.py
    +-- guardrails_eval.py
    +-- data_parser.py
    +-- app.py

    ui/
    +-- app.py

    DATA/
    +-- true_data/
    +-- noisy_data/

    processed_data/
    +-- true/
    +-- noisy/

    DOCS/
    +-- architecture and implementation notes


24. LOCAL SETUP
===============

Create a Python environment and install the dependencies:

    pip install -r requirements.txt


Create a .env file containing the required service credentials.

Core services include:

    QDRANT_API_KEY
    QDRANT_CLUSTER_ENDPOINT
    PORTKEY_API_KEY
    GROQ_API_KEY
    GEMINI_API_KEY
    LOGFIRE_TOKEN

The exact variables required depend on which embedding and gateway configuration is enabled.

Never commit real API keys to source control.


25. INGESTION
============

The ingestion processor can be executed as a module.

Examples:

    python -m app.ingestion.processor DATA --wipe

or:

    python -m app.ingestion.processor DATA/true_data true

The --wipe option recreates the Qdrant collection before ingestion.

The ingestion pipeline will:

    parse
    -> chunk
    -> save processed metadata
    -> embed
    -> index into Qdrant


26. START THE API
=================

Run the FastAPI application with Uvicorn.

Example:

    uvicorn app.main:app --reload --port 8000


The API exposes:

    GET  /
    GET  /graph
    POST /query


Example request:

    POST /query

    {
        "q": "How do I monitor a Kubernetes Job?",
        "thread_id": "demo-session"
    }


27. START THE UI
================

Run the Streamlit chat application:

    streamlit run ui/app.py


The UI provides:

    - conversational interface
    - session memory
    - visible reasoning/status steps
    - retrieved source inspection
    - backend connection
    - Logfire tracing


28. RUN THE EVALUATION SUITE
============================

The evaluation application can be started with:

    streamlit run evals/app.py


The intended workflow is:

    1. Review the golden dataset.
    2. Start the FastAPI backend.
    3. Execute the live pipeline.
    4. Run the guardrail evaluation.
    5. Run RAGAS metrics.
    6. Inspect aggregate and per-question results.
    7. Modify one pipeline component.
    8. Re-run the benchmark.
    9. Compare the results.


29. EXPERIMENTATION FRAMEWORK
=============================

The repository is designed to support controlled RAG experiments.

Potential experiment dimensions include:

    Chunking
        500 / 1000 / 1500 characters
        paragraph-based vs structure-aware

    Retrieval
        dense retrieval
        different embedding models
        different top-K values

    Reranking
        no reranker
        FlashRank
        different top-N values

    Generation
        different LLMs
        different prompts
        different context limits

    Routing
        conversational vs technical
        deterministic shortcuts vs LLM routing

    Safety
        deterministic rules
        NeMo Guardrails
        allow-listed intent validation

Each experiment should be evaluated against the same golden dataset whenever possible.


30. ENGINEERING TRADE-OFFS
==========================

The system deliberately uses multiple stages because each stage solves a different problem.

Dense retrieval:
    Fast candidate discovery.

Cross-encoder reranking:
    More precise relevance ordering.

LLM generation:
    Natural-language synthesis.

Guardrails:
    Safety and policy control.

LangGraph:
    Explicit orchestration and state management.

Portkey:
    Gateway-level model/provider abstraction and operational controls.

Logfire:
    Traceability across the pipeline.

RAGAS:
    Quality measurement.

Golden dataset:
    Regression protection.


The resulting system is more complex than a single LLM call.

That complexity is intentional.

Production RAG is not simply:

    Query -> LLM

It is a chain of independently failing components.


31. FAILURE-DRIVEN DEVELOPMENT
==============================

One of the most important outcomes of this project is the ability to learn from failure.

For example, a poor Faithfulness score does not automatically mean:

    "The LLM is bad."

It could mean:

    retrieval missed the evidence
    reranking selected poor chunks
    context construction lost information
    prompt instructions were insufficient
    generation introduced unsupported claims

Likewise, poor Context Recall points upstream toward retrieval.

This creates a debugging principle:

    Measure the failure.
        |
        v
    Locate the layer.
        |
        v
    Change one variable.
        |
        v
    Re-run the same golden benchmark.
        |
        v
    Keep or reject the change based on evidence.


32. WHAT THIS PROJECT DEMONSTRATES
==================================

This repository demonstrates practical experience with:

    - End-to-end RAG architecture
    - Multi-format document ingestion
    - Document chunking
    - Local embedding generation
    - Qdrant vector search
    - Cross-encoder reranking
    - LangGraph agent orchestration
    - Conversation memory
    - NeMo Guardrails
    - Deterministic safety checks
    - Portkey LLM gateway integration
    - Provider abstraction
    - LLM observability
    - Golden datasets
    - RAGAS evaluation
    - LLM-as-a-Judge concepts
    - Guardrail precision / recall / accuracy
    - Failure analysis
    - Performance and rate-limit considerations
    - Regression-oriented RAG experimentation


33. WHAT MAKES THIS "ENTERPRISE-GRADE"
======================================

"Enterprise-grade" is not being used here to claim production certification or a particular scale.

It refers to the engineering concerns intentionally addressed by the project:

    Reliability
        Fallback paths and controlled failures

    Observability
        Traces across application stages

    Safety
        Deterministic checks + LLM guardrails

    Scalability
        Vector database retrieval and separated pipeline stages

    Maintainability
        Modular ingestion, retrieval, agent, gateway and evaluation layers

    Testability
        Golden datasets and repeatable evaluations

    Measurability
        Retrieval, generation, safety and tool metrics

    Operational control
        Gateway abstraction, rate-limit awareness and bounded context


34. CURRENT LIMITATIONS
=======================

This is an engineering benchmark and learning project, not a claim of a fully hardened production platform.

Known areas for further work include:

    - Larger and more diverse golden datasets
    - Human-labelled evaluation calibration
    - More robust semantic/structure-aware chunking
    - Hybrid dense + sparse retrieval
    - Metadata filtering
    - Retrieval diversification
    - More reranker comparisons
    - Automated regression thresholds
    - Larger-scale load testing
    - P50/P95/P99 latency benchmarking
    - Cost-per-query measurement
    - Stronger structured guardrail outputs
    - More comprehensive adversarial evaluation
    - CI/CD integration for evaluation gates


35. FUTURE ROADMAP
==================

The next evolution of the system can focus on:

    Retrieval
        Dense + BM25 hybrid retrieval
        Query expansion
        Multi-query retrieval
        Metadata filtering

    Ranking
        Reranker benchmarking
        Dynamic top-K
        Retrieval confidence thresholds

    Generation
        Model comparison
        Prompt/version management
        Structured outputs

    Evaluation
        Larger golden benchmark
        Human calibration set
        Automated regression gates
        Model/judge comparison
        Cost and latency metrics

    Production
        P50/P95/P99 measurement
        Load testing
        CI/CD evaluation gates
        Gateway-level routing
        Better caching strategy
        Failure recovery


36. THE CORE LESSON
===================

The most important lesson from this project is that RAG quality is a system property.

A powerful LLM cannot compensate indefinitely for poor retrieval.

A strong retriever cannot compensate for an ungrounded generator.

A guardrail is not useful if its decision is not correctly interpreted by the application.

A fast system is not necessarily a good system.

And a good-looking answer is not the same thing as a correct answer.

A production-oriented RAG system therefore needs a feedback loop:

    BUILD
      |
      v
    OBSERVE
      |
      v
    EVALUATE
      |
      v
    DIAGNOSE
      |
      v
    OPTIMIZE
      |
      +--------------------+
                           |
                           v
                         REPEAT


37. CONCLUSION
==============

This project started as an implementation of Retrieval-Augmented Generation.

It evolved into something more useful:

    a measurable engineering environment for building,
    breaking, evaluating, and improving RAG systems.

The central objective is not to demonstrate that an LLM can answer a question.

The objective is to understand:

    Why did it answer that way?
    Did retrieval provide the right evidence?
    Was the answer grounded?
    Was the correct tool selected?
    Did the guardrail behave correctly?
    Where did latency occur?
    What did the change improve?
    What did it make worse?

That mindset is the foundation of reliable LLM engineering.

The next step is not to make the system look smarter.

It is to make every layer:

    more measurable,
    more explainable,
    more reliable,
    and easier to improve.


------------------------------------------------------------
PROJECT SUMMARY
------------------------------------------------------------

    Architecture:
        Agentic RAG + LangGraph

    API:
        FastAPI

    UI / Evaluation:
        Streamlit

    Vector Database:
        Qdrant

    Embeddings:
        Sentence Transformers / configurable embedding services

    Reranker:
        FlashRank

    LLM Gateway:
        Portkey

    Guardrails:
        NVIDIA NeMo Guardrails + deterministic checks

    Observability:
        Pydantic Logfire

    Evaluation:
        RAGAS + custom guardrail evaluation

    Benchmark:
        Golden dataset with live pipeline evaluation

    Primary engineering focus:
        Retrieval quality, grounding, reliability, observability and measurable iteration.


------------------------------------------------------------
END
------------------------------------------------------------
