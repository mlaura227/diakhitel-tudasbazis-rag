"""
Bounded batch evaluation of Basic RAG versus Agentic RAG.
This script deliberately DOES NOT auto-grade answer factual correctness.
Status agreement and citation overlap are diagnostics only; human review is required for false premises, nuanced conditions and citation entailment.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path
import statistics
import subprocess
import sys
import time

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATASET = Path(__file__).resolve().parent / "test_questions.jsonl"
RESULT_PREFIX = "EVAL_RESULT_JSON="

OUTPUT_FIELDS = [
    "id", "category", "question", "engine", "expected_outcome", "expected_status",
    "observed_status", "mode", "elapsed_seconds", "timed_out", "error_type", "error",
    "status_match", "gold_relevant_sections", "cited_sections", "gold_source_coverage",
    "retrieval_rounds", "rewrite_count", "retrieval_grade", "graph_trace", "answer",
    "expected_gold_facts", "grading_notes", "human_answer_correct", "human_citation_supported",
    "human_scope_safe", "human_notes",
]


def load_cases(path: Path = DATASET) -> list[dict]:
    rows = [json.loads(s) for s in path.read_text(encoding="utf-8").splitlines() if s.strip()]
    if not rows or any(not row.get("question") for row in rows):
        raise ValueError("No valid test questions")
    if len({r["id"] for r in rows}) != len(rows):
        raise ValueError("Duplicate test case IDs")
    return rows


def sections_from_sources(result: dict) -> list[str]:
    return sorted({
        f"{s.get('doc_id', '')}:{s.get('section_id', '')}"
        for s in (result.get("sources") or [])
        if s.get("doc_id") and s.get("section_id")
    })


def invoke_once(engine: str, question: str, timeout_s: float) -> dict:
    command = [
        sys.executable, "-m", "evaluation.worker", "--engine", engine,
        "--question", question,
    ]
    env = os.environ.copy()
    env["PYTHONIOENCODING"] = "utf-8"
    started = time.perf_counter()
    try:
        completed = subprocess.run(
            command, cwd=PROJECT_ROOT, env=env,
            capture_output=True, text=True, encoding="utf-8", errors="replace",
            timeout=timeout_s, check=False,
        )
    except subprocess.TimeoutExpired:
        return {"kind": "timeout", "elapsed_seconds": round(time.perf_counter() - started, 3)}
    elapsed = round(time.perf_counter() - started, 3)
    marker = next((line[len(RESULT_PREFIX):] for line in reversed(completed.stdout.splitlines())
                   if line.startswith(RESULT_PREFIX)), None)
    if marker is None:
        return {
            "kind": "error", "elapsed_seconds": elapsed,
            "error_type": "MissingResult", "error": (completed.stderr or completed.stdout)[-1200:],
        }
    try:
        payload = json.loads(marker)
    except json.JSONDecodeError as exc:
        return {"kind": "error", "elapsed_seconds": elapsed, "error_type": "BadWorkerJSON", "error": str(exc)}
    if not payload.get("ok"):
        return {
            "kind": "error", "elapsed_seconds": elapsed,
            "error_type": payload.get("error_type", "Error"), "error": payload.get("error", ""),
        }
    return {"kind": "ok", "elapsed_seconds": elapsed, "result": payload["result"]}


def make_row(case: dict, engine: str, run: dict) -> dict:
    result = run.get("result") or {}
    expected = case["expected_status"]
    observed = result.get("status") or ("timeout" if run["kind"] == "timeout" else "error")
    cited = sections_from_sources(result)
    gold = set(case.get("gold_relevant_sections") or [])
    # Important: this is an automatic *citation overlap* check only, not factual validation.
    source_coverage = (len(gold.intersection(cited)) / len(gold)) if gold else ""
    return {
        "id": case["id"], "category": case["category"], "question": case["question"],
        "engine": engine, "expected_outcome": case["expected_outcome"],
        "expected_status": expected, "observed_status": observed,
        "mode": result.get("mode", ""), "elapsed_seconds": run["elapsed_seconds"],
        "timed_out": run["kind"] == "timeout", "error_type": run.get("error_type", ""),
        "error": run.get("error", ""), "status_match": observed == expected,
        "gold_relevant_sections": "; ".join(sorted(gold)),
        "cited_sections": "; ".join(cited), "gold_source_coverage": source_coverage,
        "retrieval_rounds": result.get("retrieval_rounds", ""),
        "rewrite_count": result.get("rewrite_count", ""),
        "retrieval_grade": result.get("retrieval_grade", ""),
        "graph_trace": " -> ".join(result.get("graph_trace") or []),
        "answer": result.get("answer", ""),
        "expected_gold_facts": case["gold_facts"], "grading_notes": case["grading_notes"],
        "human_answer_correct": "", "human_citation_supported": "",
        "human_scope_safe": "", "human_notes": "",
    }


def summary(rows: list[dict]) -> str:
    lines = []
    for engine in ("basic", "agentic"):
        engine_rows = [row for row in rows if row["engine"] == engine]
        if not engine_rows:
            continue
        successful = [r for r in engine_rows if r["observed_status"] not in ("timeout", "error")]
        times = [float(r["elapsed_seconds"]) for r in successful]
        lat_med = f"{statistics.median(times):.2f}s" if times else "n/a"
        lat_p95 = f"{sorted(times)[__import__('math').ceil(.95 * len(times)) - 1]:.2f}s" if times else "n/a"
        lines.append(
            f"{engine}: {len(engine_rows)} cases, "
            f"{sum(bool(r['status_match']) for r in engine_rows)} status matches, "
            f"{sum(bool(r['timed_out']) for r in engine_rows)} timeouts, "
            f"{sum(r['observed_status']=='error' for r in engine_rows)} errors, "
            f"median completed latency {lat_med}, p95 completed latency {lat_p95}"
        )
        for category in sorted({r["category"] for r in engine_rows}):
            selection = [r for r in engine_rows if r["category"] == category]
            lines.append(
                f"  {category}: {sum(bool(r['status_match']) for r in selection)}/{len(selection)} "
                "status matches (NOT answer accuracy)"
            )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--engine", choices=("basic", "agentic", "both"), default="both")
    parser.add_argument("--timeout", type=float, default=75.0, help="Seconds per engine/question; >= 1")
    parser.add_argument("--category", help="Optional single category filter")
    parser.add_argument("--max-cases", type=int, default=None, help="First N cases; development smoke tests only")
    parser.add_argument("--dataset", type=Path, default=DATASET)
    parser.add_argument("--output", type=Path, default=PROJECT_ROOT / "evaluation" / "results.csv")
    args = parser.parse_args()
    if args.timeout < 1:
        parser.error("--timeout must be at least one second")
    if args.max_cases is not None and args.max_cases < 1:
        parser.error("--max-cases must be positive")
    cases = load_cases(args.dataset)
    if args.category:
        cases = [c for c in cases if c["category"] == args.category]
        if not cases:
            parser.error("Unknown/empty category")
    if args.max_cases is not None:
        cases = cases[:args.max_cases]
    engines = ["basic", "agentic"] if args.engine == "both" else [args.engine]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    # Flush each result so an interrupted batch leaves a useful partial CSV.
    with args.output.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=OUTPUT_FIELDS)
        writer.writeheader()
        for idx, case in enumerate(cases, 1):
            # Counterbalance run order in two-engine comparisons.
            this_order = engines if idx % 2 else list(reversed(engines))
            for engine in this_order:
                print(f"[{idx}/{len(cases)}] {case['id']} | {engine} | {case['question'][:65]}", flush=True)
                outcome = invoke_once(engine, case["question"], args.timeout)
                row = make_row(case, engine, outcome)
                rows.append(row)
                writer.writerow(row)
                f.flush()
                print(f"    {row['observed_status']} | {row['elapsed_seconds']:.1f}s", flush=True)
    print("\n" + summary(rows))
    print(f"Review file: {args.output}")
    print("Manually score human_answer_correct / human_citation_supported / human_scope_safe.")


if __name__ == "__main__":
    main()
