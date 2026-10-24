"""
Phase 1 — Live Pipeline.
Calls the running FastAPI /query endpoint for each golden sample.
Captures: actual_response (truncated to 300 chars), actual_contexts (from sources),
and actual_tools_called (detected from thought_process).
"""


import time
import copy
import json
import os
from pathlib import Path

import requests
import logfire

API_URL = "http://localhost:8000/query"
RESPONSE_TRUNCATE = 300
DELAY_BETWEEN_CALLS = 10   # seconds — stays within Groq RPM on the main key
REQUEST_TIMEOUT = 300      # seconds — guardrails + LangGraph + Groq can take >60s

def _apply_eval_config(cfg: dict | None = None) -> dict:
    """Load pipeline knobs from evaluations_configs_metrics/config.yaml."""
    global API_URL, RESPONSE_TRUNCATE, DELAY_BETWEEN_CALLS, REQUEST_TIMEOUT

    from evals.experiment_io import load_eval_config, eval_section

    full = cfg if cfg is not None else load_eval_config()
    ev = eval_section(full)
    API_URL = ev.get("api_url", API_URL)
    RESPONSE_TRUNCATE = int(ev.get("response_truncate", RESPONSE_TRUNCATE))
    DELAY_BETWEEN_CALLS = int(ev.get("delay_between_calls", DELAY_BETWEEN_CALLS))
    REQUEST_TIMEOUT = int(ev.get("request_timeout", REQUEST_TIMEOUT))
    return full


def detect_tool(thought_process: list) -> str:
    """
    Maps the thought_process list from /query response to a tool name.
    Planner sets:  'Intent: Technical' + 'Search Term: ...' → retrieve_documents
                   'Intent: Conversational/Memory'           → direct_answer
    main.py sets:  'Intent: Guardrails Fired'                → guardrails
    """
    joined = " ".join(thought_process).lower()
    if "guardrails fired" in joined:
        return "guardrails"
    if "intent: technical" in joined or "search term:" in joined or "context retrieved" in joined:
        return "retrieve_documents"
    if "conversational" in joined or "memory" in joined:
        return "direct_answer"
    return "unknown"


def run_pipeline(
    golden_dataset: dict,
    progress_callback=None,
    cfg: dict | None = None,
    run_dir: str | Path | None = None,
) -> dict:
    """
    Enriches each rag_sample in golden_dataset with live API results.
    Returns a deep copy with actual_response, actual_contexts, actual_tools_called filled.
    progress_callback(i, total, question, stage, response="") is called per step.
    API URL / truncate / delays come from evaluations_configs_metrics/config.yaml.

    If run_dir is set, writes/updates golden_vs_actualresponse.json after every response.
    """
    from evals.experiment_io import write_golden_vs_actual

    _apply_eval_config(cfg)
    dataset = copy.deepcopy(golden_dataset)
    samples = dataset["rag_samples"]
    n = len(samples)
    out_dir = Path(run_dir) if run_dir else None

    with logfire.span("🚀 Eval Phase 1 — Live Pipeline", total_samples=n):
        for i, sample in enumerate(samples):
            question = sample["question"]

            if progress_callback:
                progress_callback(i, n, question, "calling")

            with logfire.span(
                f"📤 Live Query {i + 1}/{n}",
                question=question[:80],
                domain=sample.get("domain", ""),
            ):
                try:
                    resp = requests.post(
                        API_URL,
                        json={"q": question, "thread_id": f"eval_run_{i}"},
                        timeout=REQUEST_TIMEOUT,
                    )
                    resp.raise_for_status()
                    data = resp.json()

                    raw_answer = data.get("answer") or ""
                    thought_process = data.get("thought_process") or []
                    sources = data.get("sources") or []

                    # Keep full answer for metrics.json; truncated copy for RAGAS token budget
                    sample["actual_response_full"] = raw_answer
                    sample["actual_response"] = raw_answer[:RESPONSE_TRUNCATE]
                    sample["actual_contexts"] = sources[:5]
                    sample["actual_tools_called"] = [detect_tool(thought_process)]

                    logfire.info(
                        "✅ Response captured",
                        tool=sample["actual_tools_called"][0],
                        response_chars=len(raw_answer),
                        scored_chars=len(sample["actual_response"]),
                        context_chunks=len(sources),
                    )

                except requests.exceptions.ConnectionError:
                    logfire.error("❌ Cannot reach FastAPI — is the app running on :8000?")
                    sample["actual_response_full"] = ""
                    sample["actual_response"] = ""
                    sample["actual_contexts"] = sample.get("relevant_contexts", [])
                    sample["actual_tools_called"] = ["unknown"]

                except Exception as e:
                    logfire.error(f"❌ Query failed: {e}")
                    sample["actual_response_full"] = ""
                    sample["actual_response"] = ""
                    sample["actual_contexts"] = sample.get("relevant_contexts", [])
                    sample["actual_tools_called"] = ["unknown"]

            # Flush Q/A file after every response (even empty / failed)
            if out_dir is not None:
                write_golden_vs_actual(
                    out_dir,
                    samples[: i + 1],
                    run_id=out_dir.name,
                    status="in_progress",
                )

            if progress_callback:
                progress_callback(i, n, question, "done", sample["actual_response"])

            if i < n - 1:
                time.sleep(DELAY_BETWEEN_CALLS)

        if out_dir is not None:
            write_golden_vs_actual(
                out_dir,
                samples,
                run_id=out_dir.name,
                status="pipeline_complete",
            )

    return dataset


def save_results(dataset: dict, path: str) -> None:
    with open(path, "w") as f:
        json.dump(dataset, f, indent=2)
        
        
def load_golden_dataset(cfg: dict | None = None) -> dict:
    """Load golden dataset; path can be overridden via YAML eval.golden_dataset."""
    from evals.experiment_io import ROOT, load_eval_config, eval_section

    try:
        full = cfg if cfg is not None else load_eval_config()
        rel = eval_section(full).get("golden_dataset") or "evals/golden_dataset.json"
        golden_path = Path(rel) if os.path.isabs(rel) else ROOT / rel
    except FileNotFoundError:
        golden_path = Path(__file__).parent / "golden_dataset.json"

    with open(golden_path, "r", encoding="utf-8") as f:
        return json.load(f)
