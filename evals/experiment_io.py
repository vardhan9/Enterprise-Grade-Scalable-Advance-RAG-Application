"""
Experiment config + metrics I/O.

- Loads active YAML from evaluations_configs_metrics/config.yaml
- Snapshots each completed run under baselines/<run_id>/
- Writes metrics.json (RAG scores + guardrails passes/failures)
- Appends a compact line-block to configs_and_finalmetrics.txt for analytics
"""

from __future__ import annotations

import json
import os
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parent.parent
CONFIGS_DIR = ROOT / "evaluations_configs_metrics"
ACTIVE_CONFIG_PATH = CONFIGS_DIR / "config.yaml"
BASELINES_DIR = CONFIGS_DIR / "baselines"
ANALYTICS_LOG = CONFIGS_DIR / "configs_and_finalmetrics.txt"


def load_eval_config(path: str | Path | None = None) -> dict:
    """Load the active (or given) experiment YAML. Raises if missing/invalid."""
    cfg_path = Path(path) if path else ACTIVE_CONFIG_PATH
    if not cfg_path.exists():
        raise FileNotFoundError(f"Eval config not found: {cfg_path}")
    with open(cfg_path, encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}
    if not isinstance(cfg, dict):
        raise ValueError(f"Eval config must be a mapping: {cfg_path}")
    return cfg


def save_eval_config(cfg: dict, path: str | Path | None = None) -> Path:
    """Write the active experiment YAML (preserves a short header comment)."""
    cfg_path = Path(path) if path else ACTIVE_CONFIG_PATH
    cfg_path.parent.mkdir(parents=True, exist_ok=True)
    header = (
        "# Active experiment configuration.\n"
        "# Edited from the Streamlit eval UI and/or by hand.\n"
        "# On completion, a snapshot is copied into baselines/<run_id>/config.yaml.\n\n"
    )
    with open(cfg_path, "w", encoding="utf-8") as f:
        f.write(header)
        yaml.safe_dump(cfg, f, default_flow_style=False, sort_keys=False, allow_unicode=True)
    return cfg_path


def eval_section(cfg: dict | None = None) -> dict:
    """Return the `eval` block (empty dict if absent)."""
    cfg = cfg if cfg is not None else load_eval_config()
    return cfg.get("eval") or {}


def _next_run_id(run_name: str) -> str:
    """
    YYYYMMDD_HHMMSS_<sanitized_name>_run_N (UTC).

    Time stamp sorts recent runs first when listing by name; run_N covers
    rare collisions inside the same second.
    """
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    safe = "".join(c if c.isalnum() or c in "-_" else "_" for c in (run_name or "run")).strip("_")
    safe = safe or "run"
    prefix = f"{stamp}_{safe}_run_"
    BASELINES_DIR.mkdir(parents=True, exist_ok=True)
    existing = [
        p.name for p in BASELINES_DIR.iterdir()
        if p.is_dir() and p.name.startswith(prefix)
    ]
    nums = []
    for name in existing:
        suffix = name[len(prefix):]
        if suffix.isdigit():
            nums.append(int(suffix))
    n = max(nums, default=0) + 1
    return f"{prefix}{n}"


def _avg_from_df_or_dict(metric_results: dict, key: str) -> float | None:
    obj = metric_results.get(key)
    if obj is None:
        return None
    if hasattr(obj, "mean"):
        # pandas DataFrame / Series
        col = obj[key] if key in getattr(obj, "columns", []) else obj
        val = float(col.mean())
        return round(val, 3)
    if isinstance(obj, dict) and "avg" in obj:
        return obj["avg"]
    return None


def _per_sample_scores(metric_results: dict, key: str) -> list[dict]:
    obj = metric_results.get(key)
    if obj is None:
        return []
    if hasattr(obj, "to_dict"):
        return obj.to_dict(orient="records")
    if isinstance(obj, dict) and "per_sample" in obj:
        return obj["per_sample"]
    return []


def _scored_sample_count(metric_results: dict) -> int:
    for key in (
        "faithfulness",
        "answer_relevancy",
        "context_precision",
        "context_recall",
        "answer_correctness",
        "tool_correctness",
    ):
        rows = _per_sample_scores(metric_results, key)
        if rows:
            return len(rows)
    return 0


def _enrich_dataset_info(dataset_info: dict | None, metric_results: dict) -> dict:
    info = dict(dataset_info or {})
    scored = _scored_sample_count(metric_results)
    golden = info.get("rag_samples") or info.get("rag_samples_golden")
    info["rag_samples_scored"] = scored
    if golden is not None:
        info["rag_samples_golden"] = golden
        info["rag_samples"] = golden
        if scored < int(golden):
            info.setdefault(
                "notes",
                f"Averages are over {scored} scored samples, not the full {golden} golden set. "
                "Samples with empty actual_response are skipped by _prep_samples.",
            )
    return info


def _capture_reproducibility(run_id: str, timestamp_utc: str, cfg: dict) -> dict:
    """Best-effort git + package snapshot for the run folder."""
    import subprocess
    import sys

    git: dict[str, Any] = {}
    try:
        def _git(*args: str) -> str:
            return subprocess.check_output(
                ["git", *args], cwd=ROOT, text=True, stderr=subprocess.DEVNULL
            ).strip()

        git = {
            "branch": _git("rev-parse", "--abbrev-ref", "HEAD"),
            "commit": _git("rev-parse", "HEAD"),
            "commit_short": _git("rev-parse", "--short", "HEAD"),
            "commit_message": _git("log", "-1", "--format=%s"),
            "dirty": bool(_git("status", "--porcelain")),
        }
    except Exception as e:
        git = {"error": str(e)}

    pkg_names = [
        "ragas", "instructor", "deepeval", "fastapi", "uvicorn", "streamlit",
        "langchain", "langgraph", "langchain-openai", "qdrant-client", "flashrank",
        "sentence-transformers", "nemoguardrails", "portkey-ai", "openai",
        "PyYAML", "logfire", "nest-asyncio", "pydantic",
    ]
    packages: dict[str, str] = {}
    try:
        import importlib.metadata as md
        for name in pkg_names:
            try:
                packages[name] = md.version(name)
            except Exception:
                packages[name] = "NOT_INSTALLED"
    except Exception:
        packages = {}

    chunk = cfg.get("chunking") or {}
    return {
        "run_id": run_id,
        "timestamp_utc": timestamp_utc,
        "git": git,
        "python": sys.version.split()[0],
        "packages": packages,
        "evaluation_command": {
            "backend": "uvicorn app.main:app --reload --port 8000",
            "evals_ui": "streamlit run evals/app.py",
            "steps": [
                "Start FastAPI on :8000",
                "streamlit run evals/app.py",
                "Step 2: Run Live Pipeline + guardrails",
                "Step 3: Run Eval Metrics (auto-persists baselines/<run_id>/)",
            ],
        },
        "chunk_overlap_resolution": {
            "chunk_overlap": chunk.get("chunk_overlap", 0),
            "status": "confirmed" if "chunk_overlap" in chunk else "default_0_no_overlap_in_splitter",
            "source": "app/ingestion/chunking/splitter.py",
        },
    }


def _build_samples_qa(metric_results: dict, enriched_dataset: dict | None = None) -> list[dict]:
    """
    One entry per scored sample with golden + full actual response.
    Prefer enriched_dataset (has all golden fields); fall back to metric rows.
    """
    by_id: dict[Any, dict] = {}

    if enriched_dataset:
        for s in enriched_dataset.get("rag_samples") or []:
            full = s.get("actual_response_full") or s.get("actual_response") or ""
            scored = s.get("actual_response") or ""
            if not (full or scored):
                continue
            by_id[s.get("id")] = {
                "id": s.get("id"),
                "domain": s.get("domain"),
                "question": s.get("question", ""),
                "golden_answer": s.get("reference", ""),
                "actual_full_response": full,
                #"actual_response_scored": scored,
                "actual_tools_called": s.get("actual_tools_called") or [],
                "expected_tools": s.get("expected_tools") or [],
            }

    # Attach scores from each metric onto the sample card
    for key in (
        "faithfulness",
        "answer_relevancy",
        "context_precision",
        "context_recall",
        "answer_correctness",
        "tool_correctness",
    ):
        for row in _per_sample_scores(metric_results, key):
            sid = row.get("id")
            if sid not in by_id:
                by_id[sid] = {
                    "id": sid,
                    "question": row.get("question", ""),
                    "golden_answer": row.get("golden_answer", ""),
                    "actual_full_response": row.get("actual_full_response", ""),
                    #"actual_response_scored": row.get("actual_response_scored", ""),
                    "scores": {},
                }
            by_id[sid].setdefault("scores", {})
            if key in row and row[key] is not None:
                by_id[sid]["scores"][key] = row[key]
            # Prefer fuller text from metric rows if enriched missing
            for field in ("question", "golden_answer", "actual_full_response"):
                if row.get(field) and not by_id[sid].get(field):
                    by_id[sid][field] = row[field]

    return list(by_id.values())


def build_metrics_payload(
    cfg: dict,
    metric_results: dict,
    guardrails_results: list | None,
    run_id: str,
    dataset_info: dict | None = None,
    enriched_dataset: dict | None = None,
) -> dict:
    """Assemble the metrics.json payload including guardrails passes/failures."""
    rag_metrics = {}
    for key in (
        "faithfulness",
        "answer_relevancy",
        "context_precision",
        "context_recall",
        "answer_correctness",
        "tool_correctness",
    ):
        avg = _avg_from_df_or_dict(metric_results, key)
        if avg is None and key not in metric_results:
            continue
        rag_metrics[key] = {
            "avg": avg,
            "per_sample": _per_sample_scores(metric_results, key),
        }

    guardrails: dict[str, Any] = {}
    if guardrails_results is not None:
        from evals.guardrails_eval import compute_guardrails_metrics

        gm = compute_guardrails_metrics(guardrails_results)
        passes = [r for r in guardrails_results if r.get("result") in ("TP", "TN")]
        failures = [r for r in guardrails_results if r.get("result") in ("FP", "FN")]
        guardrails = {
            **gm,
            "passes": [
                {
                    "id": r.get("id"),
                    "type": r.get("type"),
                    "result": r.get("result"),
                    "input": r.get("input"),
                    "expected_blocked": r.get("expected_blocked"),
                    "actual_blocked": r.get("actual_blocked"),
                }
                for r in passes
            ],
            "failures": [
                {
                    "id": r.get("id"),
                    "type": r.get("type"),
                    "result": r.get("result"),
                    "input": r.get("input"),
                    "expected_blocked": r.get("expected_blocked"),
                    "actual_blocked": r.get("actual_blocked"),
                    "description": r.get("description"),
                }
                for r in failures
            ],
        }

    return {
        "run_id": run_id,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "run": cfg.get("run") or {},
        "dataset": _enrich_dataset_info(dataset_info, metric_results),
        "samples": _build_samples_qa(metric_results, enriched_dataset),
        "rag_metrics": rag_metrics,
        "guardrails": guardrails,
    }


def _format_analytics_block(cfg: dict, metrics: dict, run_dir: Path) -> str:
    sys_cfg = cfg.get("system") or {}
    ret = cfg.get("retrieval") or {}
    chunk = cfg.get("chunking") or {}
    run = cfg.get("run") or {}
    ds = metrics.get("dataset") or {}
    rag = metrics.get("rag_metrics") or {}
    g = metrics.get("guardrails") or {}

    rag_parts = [
        f"{k}={v.get('avg')}" for k, v in rag.items() if v.get("avg") is not None
    ]
    pass_ids = [str(p.get("id")) for p in g.get("passes") or []]
    fail_ids = [str(f.get("id")) for f in g.get("failures") or []]

    lines = [
        "=" * 80,
        f"RUN: {metrics.get('run_id')} | {metrics.get('timestamp_utc')}",
        f"NAME: {run.get('name', '')} | NOTES: {run.get('notes', '')}",
        (
            f"CONFIG: embedding={sys_cfg.get('embedding_model')} "
            f"dim={sys_cfg.get('embedding_dim')} backend={sys_cfg.get('embedding_backend')} | "
            f"llm={sys_cfg.get('llm_model')} temp={sys_cfg.get('llm_temperature')} | "
            f"chunk={chunk.get('strategy')}/{chunk.get('chunk_size')} "
            f"overlap={chunk.get('chunk_overlap', 0)} | "
            f"retrieval_limit={ret.get('search_limit')} rerank_top_n={ret.get('rerank_top_n')} "
            f"reranker={ret.get('reranker')}"
        ),
        (
            f"DATASET: golden={ds.get('rag_samples_golden', ds.get('rag_samples', 'n/a'))} "
            f"scored={ds.get('rag_samples_scored', 'n/a')}"
        ),
        f"RAG METRICS: {' '.join(rag_parts) if rag_parts else 'n/a'}",
    ]
    if g:
        lines.append(
            f"GUARDRAILS: accuracy={g.get('accuracy')} "
            f"({g.get('correct')}/{g.get('total')}) "
            f"P={g.get('precision')} R={g.get('recall')} "
            f"TP={g.get('tp')} TN={g.get('tn')} FP={g.get('fp')} FN={g.get('fn')} | "
            f"passes={pass_ids} failures={fail_ids}"
        )
    else:
        lines.append("GUARDRAILS: n/a")
    lines.append(f"ARTIFACTS: {run_dir.as_posix()}")
    lines.append("=" * 80)
    lines.append("")
    return "\n".join(lines)


GOLDEN_VS_ACTUAL_FILENAME = "golden_vs_actualresponse.json"


def qa_entry_from_sample(sample: dict) -> dict:
    """Single Q/A card for golden_vs_actualresponse.json."""
    return {
        "id": sample.get("id"),
        "question": sample.get("question", ""),
        "golden_answer": sample.get("reference", ""),
        "actual_full_response": sample.get("actual_response_full")
        if sample.get("actual_response_full") is not None
        else (sample.get("actual_response") or ""),
        #"actual_response_scored": sample.get("actual_response") or "",
    }


def build_golden_vs_actual_payload(
    samples: list,
    *,
    run_id: str | None = None,
    status: str = "in_progress",
) -> dict:
    # Include every sample that has been processed (keys present), even if empty
    processed = [s for s in samples if "actual_response" in s or "actual_response_full" in s]
    entries = [qa_entry_from_sample(s) for s in processed]
    return {
        "run_id": run_id,
        "status": status,
        "updated_utc": datetime.now(timezone.utc).isoformat(),
        "total_samples": len(samples),
        "answered_samples": len(processed),
        "samples": entries,
    }


def write_golden_vs_actual(
    run_dir: Path,
    samples: list,
    *,
    run_id: str | None = None,
    status: str = "in_progress",
) -> Path:
    """Overwrite golden_vs_actualresponse.json (call after each live response)."""
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    rid = run_id or run_dir.name
    payload = build_golden_vs_actual_payload(samples, run_id=rid, status=status)
    path = run_dir / GOLDEN_VS_ACTUAL_FILENAME
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
    return path


def begin_experiment_run(cfg: dict | None = None) -> Path:
    """
    Create baselines/<run_id>/ at pipeline start:
    - snapshot config.yaml
    - empty golden_vs_actualresponse.json (updated after each response)
    """
    cfg = cfg if cfg is not None else load_eval_config()
    run_name = (cfg.get("run") or {}).get("name") or "run"
    run_id = _next_run_id(str(run_name))
    run_dir = BASELINES_DIR / run_id
    run_dir.mkdir(parents=True, exist_ok=False)
    if ACTIVE_CONFIG_PATH.exists():
        shutil.copy2(ACTIVE_CONFIG_PATH, run_dir / "config.yaml")
    write_golden_vs_actual(run_dir, [], run_id=run_id, status="started")
    return run_dir


def persist_experiment_results(
    metric_results: dict,
    guardrails_results: list | None = None,
    cfg: dict | None = None,
    dataset_info: dict | None = None,
    enriched_dataset: dict | None = None,
    run_dir: str | Path | None = None,
) -> Path:
    """
    Write metrics.json + reproducibility.json under baselines/<run_id>/,
    refresh golden_vs_actualresponse.json, append analytics log.

    If run_dir is provided (created at pipeline start), reuse it; otherwise create a new one.
    """
    cfg = cfg if cfg is not None else load_eval_config()

    if run_dir is not None:
        out_dir = Path(run_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        run_id = out_dir.name
        # Refresh config snapshot in case Apply happened mid-run
        if ACTIVE_CONFIG_PATH.exists():
            shutil.copy2(ACTIVE_CONFIG_PATH, out_dir / "config.yaml")
    else:
        run_name = (cfg.get("run") or {}).get("name") or "run"
        run_id = _next_run_id(str(run_name))
        out_dir = BASELINES_DIR / run_id
        out_dir.mkdir(parents=True, exist_ok=False)
        if ACTIVE_CONFIG_PATH.exists():
            shutil.copy2(ACTIVE_CONFIG_PATH, out_dir / "config.yaml")

    payload = build_metrics_payload(
        cfg=cfg,
        metric_results=metric_results,
        guardrails_results=guardrails_results,
        run_id=run_id,
        dataset_info=dataset_info,
        enriched_dataset=enriched_dataset,
    )
    with open(out_dir / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)

    if enriched_dataset:
        write_golden_vs_actual(
            out_dir,
            enriched_dataset.get("rag_samples") or [],
            run_id=run_id,
            status="complete",
        )

    repro = _capture_reproducibility(run_id, payload["timestamp_utc"], cfg)
    with open(out_dir / "reproducibility.json", "w", encoding="utf-8") as f:
        json.dump(repro, f, indent=2, ensure_ascii=False)

    ANALYTICS_LOG.parent.mkdir(parents=True, exist_ok=True)
    with open(ANALYTICS_LOG, "a", encoding="utf-8") as f:
        f.write(_format_analytics_block(cfg, payload, out_dir))

    return out_dir
