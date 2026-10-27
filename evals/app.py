# ─────────────────────────────────────────────────────────────────────────────
# CRITICAL: logfire must be configured before all other imports
# ─────────────────────────────────────────────────────────────────────────────
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dotenv import load_dotenv
load_dotenv()

import logfire
logfire.configure(token=os.getenv("LOGFIRE_TOKEN"), service_name="evals")

# ─────────────────────────────────────────────────────────────────────────────
import asyncio
import json
import nest_asyncio
import pandas as pd
import streamlit as st

nest_asyncio.apply()

from evals.pipeline import run_pipeline, load_golden_dataset
from evals.guardrails_eval import run_guardrails_eval, compute_guardrails_metrics
from evals.metrics import run_all_metrics
from evals.experiment_io import (
    load_eval_config,
    save_eval_config,
    persist_experiment_results,
    begin_experiment_run,
)
import copy

# ─────────────────────────────────────────────────────────────────────────────
# Page config
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Enterprise RAG — Eval Suite",
    page_icon="🧪",
    layout="wide",
)

# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────
SCORE_COLORS = {
    "green":  "#d4edda",
    "yellow": "#fff3cd",
    "red":    "#f8d7da",
}


def _badge(score: float) -> str:
    if score >= 0.75:
        return "🟢"
    elif score >= 0.5:
        return "🟡"
    return "🔴"


def _grade(score: float) -> str:
    if score >= 0.75:
        return "✅ Good"
    elif score >= 0.5:
        return "⚠️ Fair"
    return "❌ Poor"


def _color_score(val):
    if not isinstance(val, (int, float)):
        return ""
    if val >= 0.75:
        return f"background-color: {SCORE_COLORS['green']}"
    elif val >= 0.5:
        return f"background-color: {SCORE_COLORS['yellow']}"
    return f"background-color: {SCORE_COLORS['red']}"


def _render_metric_table(df: pd.DataFrame, metric_col: str, title: str):
    avg = df[metric_col].mean()
    st.markdown(f"**{title}** — AVG: {_badge(avg)} `{avg:.2f}` {_grade(avg)}")
    styled = df.style.applymap(_color_score, subset=[metric_col]).format({metric_col: "{:.3f}"})
    st.dataframe(styled, use_container_width=True, hide_index=True)


def _run_async(coro):
    loop = asyncio.get_event_loop()
    return loop.run_until_complete(coro)


def _ensure_options(current, options: list) -> list:
    """Keep current YAML value selectable even if not in the default list."""
    opts = list(options)
    if current is not None and current not in opts:
        opts.insert(0, current)
    return opts


def _select(label: str, current, options: list, key: str):
    opts = _ensure_options(current, options)
    idx = opts.index(current) if current in opts else 0
    return st.selectbox(label, opts, index=idx, key=key)


def _build_run_notes(system: dict, chunking: dict, retrieval: dict, extra: str = "") -> str:
    """Human-readable notes that always name the concrete embedding model."""
    parts = [
        f"embedding={system.get('embedding_model')} "
        f"({system.get('embedding_backend')}, dim={system.get('embedding_dim')})",
        f"llm={system.get('llm_model')} temp={system.get('llm_temperature')}",
        f"chunk={chunking.get('strategy')}/{chunking.get('chunk_size')} "
        f"overlap={chunking.get('chunk_overlap', 0)}",
        f"retrieve={retrieval.get('search_limit')}→{retrieval.get('rerank_top_n')} "
        f"reranker={retrieval.get('reranker')}",
    ]
    note = "; ".join(parts)
    extra = (extra or "").strip()
    # Drop vague legacy phrases that hide the real model name
    if extra and "Local MiniLM embeddings" not in extra and extra not in note:
        return f"{note} | {extra}"
    return note


# Dropdown option catalogs (current config.yaml value is always injected)
CFG_OPTIONS = {
    "embedding_backend": ["local", "gemini"],
    "embedding_model": [
        "all-MiniLM-L12-v2",
        "all-MiniLM-L6-v2",
        "models/gemini-embedding-2-preview",
    ],
    "embedding_dim": [384, 768, 3072],
    "llm_model": [
        "qwen/qwen3.8-27b",
        "openai/gpt-4o-mini",
        "google/gemma-4-31b-it:free",
        "llama-3.3-70b-versatile",
        # OpenRouter: deepseek/deepseek-r1 alone → 404 "No endpoints found"
        "deepseek/deepseek-r1-0528",
        "deepseek/deepseek-chat",
        "deepseek/deepseek-chat-v3.1",
        "deepseek/deepseek-v3.2",
        "google/gemini-3.8-flash",
    ],
    "llm_temperature": [0, 0.1, 0.2, 0.5, 0.7],
    "llm_max_tokens": [500, 1000, 2048, 4096],
    "collection": ["enterprise-rag-chunk-750-overlap-0", "enterprise-rag"],
    "chunk_strategy": ["paragraph"],
    "chunk_size": [500, 800, 1000, 1200, 1500, 2000, 3000],
    "chunk_overlap": [0, 50, 150, 100, 200],
    "search_limit": [5, 8, 10, 15, 20, 30],
    "rerank_top_n": [3, 5, 8, 10],
    "reranker": ["ms-marco-MiniLM-L-6-v2", "none"],
    "judge_model": [
        "@rag-enterprise-openrouter/openai/gpt-4o-mini",
        "@rag-enterprise-openrouter/qwen/qwen3.8-27b",
        "@marthala-groq-2/google/gemma-4-31b-it:free",
    ],
    "judge_embeddings": [
        "sentence-transformers/all-MiniLM-L6-v2",
        "sentence-transformers/all-MiniLM-L12-v2",
    ],
    "batch_size": [1, 2, 5],
    "response_truncate": [200, 300, 500, 1000, 10000],
    "context_truncate": [200, 300, 500, 1000],
    "context_limit": [1, 2, 3, 5],
    "delay_between_calls": [0, 5, 10, 15, 30],
    "request_timeout": [60, 120, 300, 600],
    "guardrails_timeout": [15, 30, 60, 120],
    "guardrails_delay": [0, 1, 2, 5],
    "golden_dataset": ["evals/golden_dataset.json", "evals/og_golden_dataset.json"],
}


# ─────────────────────────────────────────────────────────────────────────────
# Session state init
# ─────────────────────────────────────────────────────────────────────────────
if "pipeline_done" not in st.session_state:
    st.session_state.pipeline_done = False
if "enriched_dataset" not in st.session_state:
    st.session_state.enriched_dataset = None
if "guardrails_results" not in st.session_state:
    st.session_state.guardrails_results = None
if "metric_results" not in st.session_state:
    st.session_state.metric_results = None
if "pipeline_rows" not in st.session_state:
    st.session_state.pipeline_rows = []
if "last_run_dir" not in st.session_state:
    st.session_state.last_run_dir = None
if "eval_cfg" not in st.session_state:
    try:
        st.session_state.eval_cfg = load_eval_config()
    except FileNotFoundError:
        st.session_state.eval_cfg = {}
if "golden" not in st.session_state:
    st.session_state.golden = load_golden_dataset(st.session_state.eval_cfg or None)

eval_cfg = st.session_state.eval_cfg
golden = st.session_state.golden

# ─────────────────────────────────────────────────────────────────────────────
# Header + experiment config confirmation (dropdowns)
# ─────────────────────────────────────────────────────────────────────────────
st.title("🧪 Enterprise RAG — Evaluation Suite")
st.caption(
    "Confirm configuration → Step 1: Ground truth → Step 2: Live pipeline → Step 3: RAGAS metrics"
)

with st.expander("⚙️ Experiment configuration (from evaluations_configs_metrics/config.yaml)", expanded=True):
    st.caption(
        "Select settings **before** Step 2 / Step 3, then **Apply & save**. "
        "That updates `config.yaml` and syncs `LLM_MODEL` / `CHUNK_SIZE` into `.env`. "
        "**You must restart uvicorn** after changing LLM — the app loads the model at import time "
        "(otherwise Step 2 still calls the previous model, e.g. Qwen)."
    )

    run = dict(eval_cfg.get("run") or {})
    system = dict(eval_cfg.get("system") or {})
    chunking = dict(eval_cfg.get("chunking") or {})
    retrieval = dict(eval_cfg.get("retrieval") or {})
    generation = dict(eval_cfg.get("generation") or {})
    ev = dict(eval_cfg.get("eval") or {})

    st.markdown("##### Run")
    c_run1, c_run2 = st.columns([1, 2])
    run["name"] = c_run1.text_input("Run name", value=str(run.get("name") or "experiment"), key="cfg_run_name")
    run_extra = c_run2.text_input(
        "Extra notes (optional — embedding model is auto-filled on Apply)",
        value="",
        key="cfg_run_notes_extra",
        placeholder="e.g. wiped collection + re-ingested true_data",
    )

    st.markdown("##### System (app ledger)")
    s1, s2, s3 = st.columns(3)
    with s1:
        system["embedding_backend"] = _select(
            "Embedding backend", system.get("embedding_backend", "local"),
            CFG_OPTIONS["embedding_backend"], "cfg_emb_backend",
        )
        system["embedding_model"] = _select(
            "Embedding model", system.get("embedding_model", "all-MiniLM-L12-v2"),
            CFG_OPTIONS["embedding_model"], "cfg_emb_model",
        )
    with s2:
        system["embedding_dim"] = _select(
            "Embedding dim", system.get("embedding_dim", 384),
            CFG_OPTIONS["embedding_dim"], "cfg_emb_dim",
        )
        system["llm_model"] = _select(
            "LLM model", system.get("llm_model", "qwen/qwen3.8-27b"),
            CFG_OPTIONS["llm_model"], "cfg_llm_model",
        )
    with s3:
        system["llm_temperature"] = _select(
            "LLM temperature", system.get("llm_temperature", 0),
            CFG_OPTIONS["llm_temperature"], "cfg_llm_temp",
        )
        system["llm_max_tokens"] = _select(
            "LLM max_tokens", system.get("llm_max_tokens", 1000),
            CFG_OPTIONS["llm_max_tokens"], "cfg_llm_max_tokens",
        )
        system["collection"] = _select(
            "Qdrant collection", system.get("collection", "enterprise-rag-chunk-750-overlap-0"),
            CFG_OPTIONS["collection"], "cfg_collection",
        )

    st.markdown("##### Chunking (ingest → embed ledger)")
    ch1, ch2, ch3 = st.columns(3)
    with ch1:
        chunking["strategy"] = _select(
            "Chunk strategy", chunking.get("strategy", "paragraph"),
            CFG_OPTIONS["chunk_strategy"], "cfg_chunk_strategy",
        )
    with ch2:
        chunking["chunk_size"] = _select(
            "Chunk size", chunking.get("chunk_size", 1500),
            CFG_OPTIONS["chunk_size"], "cfg_chunk_size",
        )
    with ch3:
        chunking["chunk_overlap"] = _select(
            "Chunk overlap", chunking.get("chunk_overlap", 0),
            CFG_OPTIONS["chunk_overlap"], "cfg_chunk_overlap",
        )
    chunking.setdefault(
        "implementation",
        chunking.get("implementation")
        or "split on \\n\\n paragraphs; no sliding-window overlap",
    )
    chunking.setdefault(
        "source",
        chunking.get("source") or "app.config.Settings.CHUNK_SIZE → chunk_text() → embed_texts()",
    )

    st.markdown("##### Retrieval (app ledger)")
    r1, r2, r3 = st.columns(3)
    with r1:
        retrieval["search_limit"] = _select(
            "Search limit (top_k)", retrieval.get("search_limit", 15),
            CFG_OPTIONS["search_limit"], "cfg_search_limit",
        )
    with r2:
        retrieval["rerank_top_n"] = _select(
            "Rerank top_n", retrieval.get("rerank_top_n", 5),
            CFG_OPTIONS["rerank_top_n"], "cfg_rerank_top_n",
        )
    with r3:
        retrieval["reranker"] = _select(
            "Reranker", retrieval.get("reranker", "ms-marco-MiniLM-L-6-v2"),
            CFG_OPTIONS["reranker"], "cfg_reranker",
        )

    st.markdown("##### Eval runtime (applied by this UI)")
    e1, e2, e3 = st.columns(3)
    with e1:
        ev["judge_model"] = _select(
            "Judge model",
            ev.get("judge_model", "@rag-enterprise-openrouter/openai/gpt-4o-mini"),
            CFG_OPTIONS["judge_model"], "cfg_judge_model",
        )
        ev["judge_embeddings"] = _select(
            "Judge embeddings",
            ev.get("judge_embeddings", "sentence-transformers/all-MiniLM-L6-v2"),
            CFG_OPTIONS["judge_embeddings"], "cfg_judge_emb",
        )
        ev["golden_dataset"] = _select(
            "Golden dataset",
            ev.get("golden_dataset", "evals/golden_dataset.json"),
            CFG_OPTIONS["golden_dataset"], "cfg_golden",
        )
    with e2:
        ev["batch_size"] = _select(
            "Metric batch size", ev.get("batch_size", 1),
            CFG_OPTIONS["batch_size"], "cfg_batch_size",
        )
        ev["response_truncate"] = _select(
            "Response truncate", ev.get("response_truncate", 300),
            CFG_OPTIONS["response_truncate"], "cfg_resp_trunc",
        )
        ev["context_truncate"] = _select(
            "Context truncate", ev.get("context_truncate", 300),
            CFG_OPTIONS["context_truncate"], "cfg_ctx_trunc",
        )
        ev["context_limit"] = _select(
            "Context limit", ev.get("context_limit", 2),
            CFG_OPTIONS["context_limit"], "cfg_ctx_limit",
        )
    with e3:
        ev["delay_between_calls"] = _select(
            "Delay between calls (s)", ev.get("delay_between_calls", 10),
            CFG_OPTIONS["delay_between_calls"], "cfg_delay",
        )
        ev["request_timeout"] = _select(
            "Request timeout (s)", ev.get("request_timeout", 300),
            CFG_OPTIONS["request_timeout"], "cfg_req_timeout",
        )
        ev["guardrails_timeout"] = _select(
            "Guardrails timeout (s)", ev.get("guardrails_timeout", 30),
            CFG_OPTIONS["guardrails_timeout"], "cfg_gr_timeout",
        )
        ev["guardrails_delay"] = _select(
            "Guardrails delay (s)", ev.get("guardrails_delay", 2),
            CFG_OPTIONS["guardrails_delay"], "cfg_gr_delay",
        )
    ev["api_url"] = st.text_input(
        "API URL",
        value=str(ev.get("api_url") or "http://localhost:8000/query"),
        key="cfg_api_url",
    )

    generation.setdefault("response_style", generation.get("response_style") or "grounded_enterprise")
    if "max_context_chars_in_prompt" not in generation:
        generation["max_context_chars_in_prompt"] = 25000

    draft_cfg = {
        "run": run,
        "system": system,
        "chunking": chunking,
        "retrieval": retrieval,
        "generation": generation,
        "eval": ev,
    }

    b1, b2, b3 = st.columns([1, 1, 2])
    apply_btn = b1.button("✅ Apply & save to config.yaml", type="primary", width="stretch")
    reload_btn = b2.button("🔄 Reload from disk", width="stretch")

    if apply_btn:
        draft_cfg["run"]["notes"] = _build_run_notes(
            draft_cfg["system"], draft_cfg["chunking"], draft_cfg["retrieval"], extra=run_extra
        )
        save_eval_config(draft_cfg)
        st.session_state.eval_cfg = copy.deepcopy(draft_cfg)
        st.session_state.golden = load_golden_dataset(st.session_state.eval_cfg)

        # Sync live-app knobs into .env so uvicorn picks them up
        from app.config import set_env_value, settings as app_settings

        set_env_value("LLM_MODEL", draft_cfg["system"].get("llm_model") or "qwen/qwen3.8-27b")
        set_env_value("LLM_TEMPERATURE", draft_cfg["system"].get("llm_temperature", 0))
        set_env_value("LLM_MAX_TOKENS", draft_cfg["system"].get("llm_max_tokens", 1000))
        set_env_value("CHUNK_SIZE", draft_cfg["chunking"].get("chunk_size", 750))
        set_env_value("CHUNK_OVERLAP", draft_cfg["chunking"].get("chunk_overlap", 0))
        set_env_value(
            "QDRANT_COLLECTION_NAME",
            draft_cfg["system"].get("collection") or "enterprise-rag-chunk-750-overlap-0",
        )
        set_env_value("SEARCH_LIMIT", draft_cfg["retrieval"].get("search_limit", 15))
        set_env_value("RERANK_TOP_N", draft_cfg["retrieval"].get("rerank_top_n", 5))

        st.success(
            f"Saved config.yaml + `.env` — "
            f"LLM=`{draft_cfg['system'].get('llm_model')}` · "
            f"retrieve `{draft_cfg['retrieval'].get('search_limit')}→"
            f"{draft_cfg['retrieval'].get('rerank_top_n')}` · "
            f"Portkey `{app_settings.portkey_chat_model()}`"
        )
        st.warning(
            "Restart FastAPI only if you changed the **LLM** (cached at import). "
            "Search limit / rerank top_n reload from `.env` on each retrieve call."
        )
        st.rerun()

    if reload_btn:
        st.session_state.eval_cfg = load_eval_config()
        st.session_state.golden = load_golden_dataset(st.session_state.eval_cfg)
        st.info("Reloaded config.yaml from disk.")
        st.rerun()

    # Confirmation strip — reflects dropdown selection; Apply writes it to disk + session
    st.markdown(
        f"**Selected (Apply & save to lock in):** `{run.get('name')}` · "
        f"embed `{system.get('embedding_model')}` ({system.get('embedding_backend')}) · "
        f"llm `{system.get('llm_model')}` · "
        f"chunk `{chunking.get('chunk_size')}`/`{chunking.get('chunk_overlap')}` · "
        f"retrieve `{retrieval.get('search_limit')}`→`{retrieval.get('rerank_top_n')}` · "
        f"judge `{ev.get('judge_model')}`"
    )
    applied = st.session_state.eval_cfg or {}
    a_run = (applied.get("run") or {}).get("name", "?")
    a_judge = (applied.get("eval") or {}).get("judge_model", "?")
    st.caption(
        f"Currently applied for pipeline/metrics: run=`{a_run}` · judge=`{a_judge}` · "
        f"source=`evaluations_configs_metrics/config.yaml`"
    )

# Keep a single source of truth for the rest of the page
eval_cfg = st.session_state.eval_cfg
golden = st.session_state.golden
st.divider()

# ─────────────────────────────────────────────────────────────────────────────
# Tabs
# ─────────────────────────────────────────────────────────────────────────────
tab1, tab2, tab3 = st.tabs(
    ["📋 Step 1 — Ground Truth", "🚀 Step 2 — Live Pipeline", "📊 Step 3 — Eval Metrics"]
)


# ═════════════════════════════════════════════════════════════════════════════
# TAB 1 — Ground Truth
# ═════════════════════════════════════════════════════════════════════════════
with tab1:
    st.subheader("Ground Truth Dataset")
    st.markdown(
        "These are the **golden Q&A pairs** built by parsing your real enterprise documents. "
        "Each entry has a question, a reference answer (ground truth), and the expected tool the RAG agent should call."
    )

    rag_rows = []
    for s in golden["rag_samples"]:
        rag_rows.append({
            "ID": s["id"],
            "Domain": s["domain"].replace("_", " ").title(),
            "Question": s["question"],
            "Reference Answer": s["reference"][:120] + "..." if len(s["reference"]) > 120 else s["reference"],
            "Expected Tool": s["expected_tools"][0] if s["expected_tools"] else "—",
        })
    df_golden = pd.DataFrame(rag_rows)
    st.dataframe(df_golden, use_container_width=True, hide_index=True)
    st.caption(f"✅ {len(rag_rows)} golden RAG samples from 5 enterprise docs")

    st.divider()

    st.subheader("Guardrails Test Cases")
    st.markdown(
        "These inputs test whether the safety rails correctly **block adversarial inputs** "
        "and **let through legitimate questions**."
    )

    g_rows = []
    for g in golden["guardrails_samples"]:
        expected_label = "🛡️ Block" if g["expected_blocked"] else "✅ Pass"
        g_rows.append({
            "ID": g["id"],
            "Input": g["input"],
            "Expected": expected_label,
            "Type": g["type"],
            "Description": g["description"],
        })
    st.dataframe(pd.DataFrame(g_rows), use_container_width=True, hide_index=True)
    st.caption("6 guardrails test cases: 3 adversarial (should block) + 3 legit (should pass)")

    with st.expander("View raw golden_dataset.json"):
        st.json(golden)


# ═════════════════════════════════════════════════════════════════════════════
# TAB 2 — Live Pipeline
# ═════════════════════════════════════════════════════════════════════════════
with tab2:
    st.subheader("Live Pipeline — Collect Real Responses")
    st.markdown(
        "Sends each golden question to your **running FastAPI app** (`localhost:8000/query`). "
        "Captures the actual response, retrieved contexts, and tool called. "
        "Full answers are kept for `metrics.json`; RAGAS scoring still uses "
        f"`response_truncate` ({(eval_cfg.get('eval') or {}).get('response_truncate', 300)} chars) to save judge tokens."
    )
    st.info(
        "⚠️ Make sure your FastAPI backend is running first: `uvicorn app.main:app --reload --port 8000`",
        icon="⚠️",
    )

    col_p1, col_p2, col_p3 = st.columns([1, 1, 2])
    run_pipeline_btn = col_p1.button(
        "▶️ Run Live Pipeline",
        type="primary",
        width="stretch",
        disabled=st.session_state.pipeline_done,
    )
    reset_btn = col_p2.button(
        "🔄 Reset & Re-run",
        width="stretch",
        disabled=not st.session_state.pipeline_done,
    )

    if reset_btn:
        st.session_state.pipeline_done = False
        st.session_state.enriched_dataset = None
        st.session_state.guardrails_results = None
        st.session_state.metric_results = None
        st.session_state.pipeline_rows = []
        st.session_state.last_run_dir = None
        st.rerun()

    if run_pipeline_btn:
        st.session_state.pipeline_rows = []
        progress_bar = st.progress(0, text="Starting pipeline...")
        live_table_slot = st.empty()
        status_slot = st.empty()

        # Create run folder immediately so golden_vs_actualresponse.json can update live
        run_dir = begin_experiment_run(eval_cfg or None)
        st.session_state.last_run_dir = str(run_dir)
        status_slot.info(f"📁 Run folder: `{run_dir}` — updating `golden_vs_actualresponse.json` after each answer")

        def pipeline_cb(i, total, question, stage, response=""):
            pct = int((i / total) * 100)
            if stage == "calling":
                progress_bar.progress(pct, text=f"[{i+1}/{total}] Calling /query: {question[:60]}...")
            else:
                short_q = question[:55] + "..." if len(question) > 55 else question
                short_r = response[:80] + "..." if len(response) > 80 else response
                st.session_state.pipeline_rows.append({
                    "#": i + 1,
                    "Question": short_q,
                    "Live Response (truncated)": short_r if short_r else "⚠️ No response",
                    "Status": "✅" if short_r else "❌",
                })
                live_table_slot.dataframe(
                    pd.DataFrame(st.session_state.pipeline_rows),
                    use_container_width=True,
                    hide_index=True,
                )
                progress_bar.progress(
                    int(((i + 1) / total) * 100),
                    text=f"[{i+1}/{total}] ✅ Done — wrote golden_vs_actualresponse.json",
                )

        with logfire.span("🚀 Streamlit — Run Pipeline Button"):
            enriched = run_pipeline(
                golden,
                progress_callback=pipeline_cb,
                cfg=eval_cfg or None,
                run_dir=run_dir,
            )
            st.session_state.enriched_dataset = enriched

        progress_bar.progress(100, text="✅ All responses collected!")
        status_slot.success(
            f"💾 {len(enriched['rag_samples'])} responses stored · "
            f"`{run_dir / 'golden_vs_actualresponse.json'}`"
        )
        # ── Guardrails tests ──────────────────────────────────────────────────
        st.divider()
        st.subheader("Guardrails Tests")
        g_progress = st.progress(0, text="Running guardrails tests...")
        g_status_slot = st.empty()

        def g_cb(i, total, input_text):
            g_progress.progress(
                int((i / total) * 100),
                text=f"[{i+1}/{total}] Testing: {input_text[:60]}...",
            )

        with logfire.span("🛡️ Streamlit — Guardrails Tests"):
            g_results = run_guardrails_eval(
                enriched["guardrails_samples"],
                progress_callback=g_cb,
                cfg=eval_cfg or None,
            )
            g_metrics = compute_guardrails_metrics(g_results)
            st.session_state.guardrails_results = g_results
            st.session_state.pipeline_done = True

        g_progress.progress(100, text="✅ Guardrails tests complete!")

        g_rows_live = []
        for r in g_results:
            result_label = {
                "TP": "🛡️ Blocked ✅", "TN": "✅ Passed ✅",
                "FP": "🛡️ Blocked ❌ (False Positive)", "FN": "✅ Passed ❌ (Missed)",
            }.get(r["result"], r["result"])
            g_rows_live.append({
                "ID": r["id"],
                "Input": r["input"][:70],
                "Expected": "🛡️ Block" if r["expected_blocked"] else "✅ Pass",
                "Actual": "Blocked" if r["actual_blocked"] else "Passed",
                "Result": result_label,
            })
        st.dataframe(pd.DataFrame(g_rows_live), use_container_width=True, hide_index=True)

        mc1, mc2, mc3, mc4 = st.columns(4)
        mc1.metric("Correct", f"{g_metrics['correct']}/{g_metrics['total']}")
        mc2.metric("Precision", f"{g_metrics['precision']:.2f}")
        mc3.metric("Recall", f"{g_metrics['recall']:.2f}")
        mc4.metric("Accuracy", f"{g_metrics['accuracy']:.2f}")

    elif st.session_state.pipeline_done:
        st.success("✅ Pipeline already run. See results below.")

        resp_rows = []
        for s in st.session_state.enriched_dataset["rag_samples"]:
            resp_rows.append({
                "#": s["id"],
                "Domain": s["domain"].replace("_", " ").title(),
                "Question": s["question"][:60],
                "Live Response": s["actual_response"][:100] + "..." if len(s.get("actual_response","")) > 100 else s.get("actual_response",""),
                "Tool Called": s["actual_tools_called"][0] if s.get("actual_tools_called") else "—",
                "Contexts Retrieved": len(s.get("actual_contexts", [])),
            })
        st.dataframe(pd.DataFrame(resp_rows), use_container_width=True, hide_index=True)

        with st.expander("View golden answer vs full actual response"):
            for s in st.session_state.enriched_dataset["rag_samples"]:
                full = s.get("actual_response_full") or s.get("actual_response") or ""
                st.markdown(f"**#{s['id']} — {s['question']}**")
                st.markdown("*Golden answer:*")
                st.write(s.get("reference") or "—")
                st.markdown("*Actual full response:*")
                st.write(full if full else "⚠️ No response")
                st.caption(
                    f"Scored/truncated ({len(s.get('actual_response') or '')} chars): "
                    f"`{(s.get('actual_response') or '')[:120]}...`"
                    if s.get("actual_response")
                    else "Scored: empty"
                )
                st.divider()

        if st.session_state.guardrails_results:
            st.divider()
            st.subheader("Guardrails Results (from previous run)")
            g_rows_prev = []
            for r in st.session_state.guardrails_results:
                result_label = {
                    "TP": "🛡️ Blocked ✅", "TN": "✅ Passed ✅",
                    "FP": "Blocked ❌ FP", "FN": "Passed ❌ FN",
                }.get(r["result"], r["result"])
                g_rows_prev.append({
                    "ID": r["id"],
                    "Input": r["input"][:70],
                    "Result": result_label,
                })
            st.dataframe(pd.DataFrame(g_rows_prev), use_container_width=True, hide_index=True)
            gm = compute_guardrails_metrics(st.session_state.guardrails_results)
            mc1, mc2, mc3, mc4 = st.columns(4)
            mc1.metric("Correct", f"{gm['correct']}/{gm['total']}")
            mc2.metric("Precision", f"{gm['precision']:.2f}")
            mc3.metric("Recall", f"{gm['recall']:.2f}")
            mc4.metric("Accuracy", f"{gm['accuracy']:.2f}")


# ═════════════════════════════════════════════════════════════════════════════
# TAB 3 — Eval Metrics
# ═════════════════════════════════════════════════════════════════════════════
with tab3:
    st.subheader("Eval Metrics — RAGAS + Tool Correctness")

    if not st.session_state.pipeline_done:
        st.warning("⚠️ Complete Step 2 (Live Pipeline) first to collect responses.")
    else:
        st.markdown(
            "Runs all **6 metric experiments** on the stored responses. "
            "Knobs come from `evaluations_configs_metrics/config.yaml`."
        )

        run_metrics_btn = st.button(
            "▶️ Run Eval Metrics",
            type="primary",
            disabled=not st.session_state.pipeline_done,
        )

        if run_metrics_btn:
            status_slot = st.empty()
            results_slots = {}

            metric_display_names = {
                "faithfulness":      "Exp 1 — Faithfulness",
                "answer_relevancy":  "Exp 2 — Answer Relevancy",
                "context_precision": "Exp 3 — Context Precision",
                "context_recall":    "Exp 4 — Context Recall",
                "answer_correctness":"Exp 5 — Answer Correctness",
                "tool_correctness":  "Exp 6 — Tool Correctness",
            }
            for key, title in metric_display_names.items():
                results_slots[key] = st.empty()

            def status_cb(msg: str):
                status_slot.info(msg)

            with logfire.span("📊 Streamlit — Run Metrics Button"):
                metric_results = _run_async(
                    run_all_metrics(
                        st.session_state.enriched_dataset,
                        status_cb=status_cb,
                        cfg=eval_cfg or None,
                    )
                )
                st.session_state.metric_results = metric_results

                enriched = st.session_state.enriched_dataset or {}
                dataset_info = {
                    "rag_samples": len(enriched.get("rag_samples") or []),
                    "guardrails_samples": len(enriched.get("guardrails_samples") or []),
                    "golden_path": (eval_cfg.get("eval") or {}).get(
                        "golden_dataset", "evals/golden_dataset.json"
                    ),
                }
                run_dir = persist_experiment_results(
                    metric_results=metric_results,
                    guardrails_results=st.session_state.guardrails_results,
                    cfg=eval_cfg or None,
                    dataset_info=dataset_info,
                    enriched_dataset=enriched,
                    run_dir=st.session_state.last_run_dir,
                )
                st.session_state.last_run_dir = str(run_dir)

            status_slot.success(
                f"✅ All 6 experiments complete! Saved → `{st.session_state.last_run_dir}`"
            )

            for key, title in metric_display_names.items():
                if key in metric_results:
                    with results_slots[key].container():
                        _render_metric_table(metric_results[key], key, title)

        elif st.session_state.metric_results:
            st.success("✅ Metrics already computed. Showing results below.")
            metric_display_names = {
                "faithfulness":      "Exp 1 — Faithfulness",
                "answer_relevancy":  "Exp 2 — Answer Relevancy",
                "context_precision": "Exp 3 — Context Precision",
                "context_recall":    "Exp 4 — Context Recall",
                "answer_correctness":"Exp 5 — Answer Correctness",
                "tool_correctness":  "Exp 6 — Tool Correctness",
            }
            for key, title in metric_display_names.items():
                if key in st.session_state.metric_results:
                    _render_metric_table(st.session_state.metric_results[key], key, title)

        # ── Final Summary ─────────────────────────────────────────────────────
        if st.session_state.metric_results:
            st.divider()
            st.subheader("Final Summary")

            mr = st.session_state.metric_results
            summary = [
                ("Faithfulness",       mr.get("faithfulness",      pd.DataFrame()).get("faithfulness",      pd.Series()).mean()),
                ("Answer Relevancy",   mr.get("answer_relevancy",  pd.DataFrame()).get("answer_relevancy",  pd.Series()).mean()),
                ("Context Precision",  mr.get("context_precision", pd.DataFrame()).get("context_precision", pd.Series()).mean()),
                ("Context Recall",     mr.get("context_recall",    pd.DataFrame()).get("context_recall",    pd.Series()).mean()),
                ("Answer Correctness", mr.get("answer_correctness",pd.DataFrame()).get("answer_correctness",pd.Series()).mean()),
                ("Tool Correctness",   mr.get("tool_correctness",  pd.DataFrame()).get("tool_correctness",  pd.Series()).mean()),
            ]

            cols = st.columns(len(summary))
            for col, (name, score) in zip(cols, summary):
                if pd.notna(score):
                    col.metric(
                        label=name,
                        value=f"{score:.2f}",
                        delta=_grade(score),
                    )

            if st.session_state.guardrails_results:
                gm = compute_guardrails_metrics(st.session_state.guardrails_results)
                st.metric(
                    label="🛡️ Guardrails Accuracy",
                    value=f"{gm['correct']}/{gm['total']}",
                    delta=f"Precision {gm['precision']:.2f} | Recall {gm['recall']:.2f}",
                )

            summary_df = pd.DataFrame([
                {"Metric": name, "Score": f"{score:.3f}" if pd.notna(score) else "—", "Grade": _grade(score) if pd.notna(score) else "—"}
                for name, score in summary
            ])
            st.dataframe(summary_df, use_container_width=True, hide_index=True)

            if st.session_state.last_run_dir:
                st.caption(
                    f"📁 Run artifacts: `{st.session_state.last_run_dir}` "
                    f"(config.yaml · metrics.json · golden_vs_actualresponse.json · reproducibility.json) · "
                    f"appended to `evaluations_configs_metrics/configs_and_finalmetrics.txt`"
                )
