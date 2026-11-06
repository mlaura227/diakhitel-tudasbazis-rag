"""
Pure presentation and export helpers for the local Streamlit RAG UI.
These helpers deliberately do not import Streamlit or run any model, so they can be tested without installing optional UI/model dependencies.
"""
from __future__ import annotations

import csv
import io
import json
from typing import Any

MODE_NAMES = {
    "basic": "Basic RAG",
    "agentic": "Agentic RAG",
    "compare": "Compare both",
}

STATUS_NAMES = {
    "answered": "Megválaszolva",
    "partial": "Részleges válasz",
    "insufficient": "Nem állapítható meg",
}


def source_line(source: dict[str, Any]) -> str:
    sid = source.get("citation_id", "?")
    title = source.get("document_title", "Ismeretlen dokumentum")
    page = source.get("page", "?")
    section = source.get("section_id", "?")
    return f"[{sid}] {title} | {page}. oldal | {section}"


def status_label(status: str) -> str:
    return STATUS_NAMES.get(status, status or "ismeretlen")


def result_summary(result: dict[str, Any]) -> str:
    return " · ".join(
        [
            status_label(str(result.get("status", ""))),
            str(result.get("mode") or "ismeretlen mód"),
            f"{len(result.get('sources') or [])} forrás",
        ]
    )


def history_as_csv(history: list[dict[str, Any]]) -> str:
    """Export recorded questions and metadata for review (not gold scores)."""
    output = io.StringIO(newline="")
    writer = csv.writer(output)
    writer.writerow(
        [
            "question",
            "engine",
            "status",
            "answer",
            "mode",
            "elapsed_seconds",
            "source_sections",
            "graph_trace",
            "retrieval_rounds",
            "rewrite_count",
            "retrieval_grade",
        ]
    )
    for turn in history:
        for engine, result in (turn.get("results") or {}).items():
            sources = result.get("sources") or []
            ids = [
                f"{src.get('doc_id', '')}:{src.get('section_id', '')}"
                for src in sources
            ]
            writer.writerow(
                [
                    turn.get("question", ""),
                    engine,
                    result.get("status", ""),
                    result.get("answer", ""),
                    result.get("mode", ""),
                    (turn.get("times") or {}).get(engine, ""),
                    "; ".join(ids),
                    " -> ".join(result.get("graph_trace") or []),
                    result.get("retrieval_rounds", ""),
                    result.get("rewrite_count", ""),
                    result.get("retrieval_grade", ""),
                ]
            )
    return output.getvalue()


def diagnostic_json(result: dict[str, Any]) -> str:
    """Diagnostic output intentionally excludes full retrieved passages."""
    keys = (
        "status", "mode", "attempts", "diagnostics", "retrieval_trace",
        "graph_trace", "retrieval_rounds", "rewrite_count",
        "retrieval_grade", "retrieval_grade_reason",
    )
    return json.dumps({k: result[k] for k in keys if k in result}, ensure_ascii=False, indent=2)
