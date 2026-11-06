"""Evaluate single-round BGE-M3 + Chroma retrieval on the curated gold refs.

This is a separate, shared initial-retrieval benchmark; NOT a claim about
LangGraph's final merged hits after rewrites. Empty-reference out-of-scope
cases are excluded from Precision/Recall calculations.
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path
from statistics import mean

from evaluation.evaluate import load_cases, DATASET, PROJECT_ROOT


def score(gold: list[str], predicted: list[str], k: int) -> dict | None:
    if not gold:
        return None
    gold_set, top = set(gold), predicted[:k]
    rel = [int(pred in gold_set) for pred in top]
    return {
        "precision_at_k": sum(rel) / k,
        "recall_at_k": len(set(top).intersection(gold_set)) / len(gold_set),
        "hit_at_1": bool(top and top[0] in gold_set),
        "mrr_at_k": next((1 / (i+1) for i, pred in enumerate(top) if pred in gold_set), 0),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--k", type=int, default=5)
    parser.add_argument("--dataset", type=Path, default=DATASET)
    parser.add_argument("--output", type=Path, default=PROJECT_ROOT / "evaluation" / "retrieval_results.csv")
    args = parser.parse_args()
    if args.k < 1:
        parser.error("--k must be >= 1")
    from src.retrieval import retrieve_documents
    rows = []
    for case in load_cases(args.dataset):
        hits = retrieve_documents(case["question"], k=args.k)
        predicted = [f"{h['metadata'].get('doc_id','')}:{h['metadata'].get('section_id','')}" for h in hits]
        metrics = score(case["gold_relevant_sections"], predicted, args.k)
        row = {"id":case["id"], "category":case["category"],
               "question":case["question"], "gold": "; ".join(case["gold_relevant_sections"]),
               "retrieved": "; ".join(predicted), "k": args.k, **(metrics or {})}
        rows.append(row)
        print(f"{case['id']}: " + (f"R@{args.k}={metrics['recall_at_k']:.2f}" if metrics else "no gold refs (exclude from P/R)"))
    args.output.parent.mkdir(exist_ok=True, parents=True)
    with args.output.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader();writer.writerows(rows)
    valid = [r for r in rows if isinstance(r.get("recall_at_k"), (float,int))]
    print(f"Measured {len(valid)}/{len(rows)} evidence-addressable questions. Gold refs are minimal, not exhaustively labelled.")
    for metric in ("precision_at_k", "recall_at_k", "hit_at_1", "mrr_at_k"):
        print(f"{metric}: {mean(r[metric] for r in valid):.3f}" if valid else f"{metric}: n/a")
    print("Note: initial retrieval only, not agentic final retrieval; P@k is a conservative proxy with incomplete relevance labels.")
    print(f"Saved: {args.output}")


if __name__ == "__main__":
    main()
