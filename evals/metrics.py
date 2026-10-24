"""
Phase 2 — RAGAS + Tool Correctness metrics.
Contexts are truncated so a single RAGAS request stays manageable.
"""

import logfire
import pandas as pd
from openai import AsyncOpenAI

from portkey_ai import PORTKEY_GATEWAY_URL
from ragas.llms import llm_factory
from ragas.embeddings import HuggingFaceEmbeddings
from ragas.metrics.collections import (
    Faithfulness,
    AnswerRelevancy,
    ContextPrecision,
    ContextRecall,
    AnswerCorrectness,
)

# Defaults — overridden by evaluations_configs_metrics/config.yaml when present
JUDGE_MODEL = "@rag-enterprise-openrouter/openai/gpt-4o-mini"
JUDGE_EMBEDDINGS = "sentence-transformers/all-MiniLM-L6-v2"
GENERAL_BATCH_SIZE = 1
CONTEXT_TRUNCATE = 300
CONTEXT_LIMIT = 2


def _apply_eval_config(cfg: dict | None = None) -> dict:
    """Load YAML eval knobs into module-level settings. Returns the full config."""
    global JUDGE_MODEL, JUDGE_EMBEDDINGS, GENERAL_BATCH_SIZE, CONTEXT_TRUNCATE, CONTEXT_LIMIT

    from evals.experiment_io import load_eval_config, eval_section

    full = cfg if cfg is not None else load_eval_config()
    ev = eval_section(full)
    JUDGE_MODEL = ev.get("judge_model", JUDGE_MODEL)
    JUDGE_EMBEDDINGS = ev.get("judge_embeddings", JUDGE_EMBEDDINGS)
    GENERAL_BATCH_SIZE = int(ev.get("batch_size", GENERAL_BATCH_SIZE))
    CONTEXT_TRUNCATE = int(ev.get("context_truncate", CONTEXT_TRUNCATE))
    CONTEXT_LIMIT = int(ev.get("context_limit", CONTEXT_LIMIT))
    return full


def _build_judge():
    client = AsyncOpenAI(
        api_key="wxhIXA6ec1jTAWaHO0qVJ96+phaN",
        base_url=PORTKEY_GATEWAY_URL,
        default_headers={
            "x-portkey-api-key": "wxhIXA6ec1jTAWaHO0qVJ96+phaN",
            "x-portkey-metadata": '{"feature":"rag-evals","_user":"RAG","environment":"dev"}',
        },
    )

    llm = llm_factory(
        JUDGE_MODEL,
        provider="openai",
        client=client,
        max_tokens=16384,
    )

    embeddings = HuggingFaceEmbeddings(
        model=JUDGE_EMBEDDINGS,
        use_api=False,
    )

    return llm, embeddings


def _prep_samples(golden_dataset: dict) -> list:
    """
    Returns only samples with actual_response populated.
    Truncates contexts to CONTEXT_TRUNCATE chars and limits to CONTEXT_LIMIT chunks.
    """
    valid = []
    for s in golden_dataset["rag_samples"]:
        response = s.get("actual_response", "").strip()
        if not response:
            continue
        raw_contexts = s.get("actual_contexts") or s.get("relevant_contexts") or []
        contexts = [c[:CONTEXT_TRUNCATE] for c in raw_contexts[:CONTEXT_LIMIT]]
        valid.append({**s, "actual_contexts": contexts})
    return valid


def _sample_meta(s: dict) -> dict:
    """Fields written into metrics.json per_sample for debugging / analysis."""
    return {
        "id": s.get("id"),
        "question": s.get("question", ""),
        "golden_answer": s.get("reference", ""),
        "actual_full_response": s.get("actual_response_full") or s.get("actual_response", ""),
        #"actual_response_scored": s.get("actual_response", ""),  # truncated for RAGAS
    }


def _score_df(metric_key: str, samples: list, scores) -> pd.DataFrame:
    return pd.DataFrame([
        {**_sample_meta(s), metric_key: round(float(r.value), 3)}
        for s, r in zip(samples, scores)
    ])


async def _batched_score(metric, inputs: list, samples: list, status_cb=None, label: str = "") -> list:
    """Runs abatch_score in chunks of GENERAL_BATCH_SIZE."""
    all_scores = []
    batches = [inputs[i : i + GENERAL_BATCH_SIZE] for i in range(0, len(inputs), GENERAL_BATCH_SIZE)]
    for b_idx, batch in enumerate(batches):
        if status_cb and label:
            status_cb(f"{label} — batch {b_idx + 1}/{len(batches)}")
        scores = await metric.abatch_score(batch)
        all_scores.extend(scores)
    return all_scores


async def run_all_metrics(golden_dataset: dict, status_cb=None, cfg: dict | None = None) -> dict:
    """
    Runs all 6 experiments. Returns dict keyed by metric name → DataFrame.
    status_cb(message: str) is called for live UI updates.
    Knobs (truncate, judge model, batch size) come from evaluations_configs_metrics/config.yaml.
    """
    _apply_eval_config(cfg)
    judge_llm, ragas_embeddings = _build_judge()
    samples = _prep_samples(golden_dataset)

    if not samples:
        raise ValueError("No samples with actual_response found. Run Phase 1 first.")

    results = {}

    with logfire.span("🧪 Eval Phase 2 — All Metrics", total_samples=len(samples)):

        # ── Exp 1: Faithfulness ───────────────────────────────────────────────
        if status_cb:
            status_cb(f"🧪 Exp 1/6 — Faithfulness ({len(samples)} samples)...")
        with logfire.span("🧪 Exp 1 — Faithfulness"):
            inputs = [
                {
                    "user_input": s["question"],
                    "response": s["actual_response"],
                    "retrieved_contexts": s["actual_contexts"],
                }
                for s in samples
            ]
            scores = await _batched_score(Faithfulness(llm=judge_llm), inputs, samples, status_cb, "Faithfulness")
            df = _score_df("faithfulness", samples, scores)
            results["faithfulness"] = df
            logfire.info("🧪 Faithfulness done", avg=round(df["faithfulness"].mean(), 3))

        # ── Exp 2: Answer Relevancy ───────────────────────────────────────────
        if status_cb:
            status_cb(f"🧪 Exp 2/6 — Answer Relevancy ({len(samples)} samples)...")
        with logfire.span("🧪 Exp 2 — Answer Relevancy"):
            inputs = [
                {"user_input": s["question"], "response": s["actual_response"]}
                for s in samples
            ]
            scores = await _batched_score(
                AnswerRelevancy(llm=judge_llm, embeddings=ragas_embeddings),
                inputs, samples, status_cb, "Answer Relevancy"
            )
            df = _score_df("answer_relevancy", samples, scores)
            results["answer_relevancy"] = df
            logfire.info("🧪 Answer Relevancy done", avg=round(df["answer_relevancy"].mean(), 3))

        # ── Exp 3: Context Precision ──────────────────────────────────────────
        if status_cb:
            status_cb(f"🧪 Exp 3/6 — Context Precision ({len(samples)} samples)...")
        with logfire.span("🧪 Exp 3 — Context Precision"):
            inputs = [
                {
                    "user_input": s["question"],
                    "reference": s["reference"],
                    "retrieved_contexts": s["actual_contexts"],
                }
                for s in samples
            ]
            scores = await _batched_score(ContextPrecision(llm=judge_llm), inputs, samples, status_cb, "Context Precision")
            df = _score_df("context_precision", samples, scores)
            results["context_precision"] = df
            logfire.info("🧪 Context Precision done", avg=round(df["context_precision"].mean(), 3))

        # ── Exp 4: Context Recall ─────────────────────────────────────────────
        if status_cb:
            status_cb(f"🧪 Exp 4/6 — Context Recall ({len(samples)} samples)...")
        with logfire.span("🧪 Exp 4 — Context Recall"):
            inputs = [
                {
                    "user_input": s["question"],
                    "reference": s["reference"],
                    "retrieved_contexts": s["actual_contexts"],
                }
                for s in samples
            ]
            scores = await _batched_score(ContextRecall(llm=judge_llm), inputs, samples, status_cb, "Context Recall")
            df = _score_df("context_recall", samples, scores)
            results["context_recall"] = df
            logfire.info("🧪 Context Recall done", avg=round(df["context_recall"].mean(), 3))

        # ── Exp 5: Answer Correctness ─────────────────────────────────────────
        if status_cb:
            status_cb(f"🧪 Exp 5/6 — Answer Correctness ({len(samples)} samples)...")
        with logfire.span("🧪 Exp 5 — Answer Correctness"):
            inputs = [
                {
                    "user_input": s["question"],
                    "response": s["actual_response"],
                    "reference": s["reference"],
                }
                for s in samples
            ]
            all_scores = await _batched_score(
                AnswerCorrectness(llm=judge_llm, embeddings=ragas_embeddings),
                inputs, samples, status_cb, "Answer Correctness"
            )
            df = _score_df("answer_correctness", samples, all_scores)
            results["answer_correctness"] = df
            logfire.info("🧪 Answer Correctness done", avg=round(df["answer_correctness"].mean(), 3))

        # ── Exp 6: Tool Correctness (no LLM — Jaccard) ───────────────────────
        if status_cb:
            status_cb("⚡ Exp 6/6 — Tool Correctness (zero LLM calls)...")
        with logfire.span("🧪 Exp 6 — Tool Correctness"):
            tool_rows = []
            for s in samples:
                called = set(s.get("actual_tools_called") or [])
                expected = set(s.get("expected_tools") or [])
                union = len(called | expected)
                score = len(called & expected) / union if union > 0 else 0.0
                tool_rows.append({**_sample_meta(s), "tool_correctness": round(score, 3)})
            df = pd.DataFrame(tool_rows)
            results["tool_correctness"] = df
            logfire.info("🧪 Tool Correctness done", avg=round(df["tool_correctness"].mean(), 3))

        if status_cb:
            status_cb("✅ All 6 experiments complete!")

    return results