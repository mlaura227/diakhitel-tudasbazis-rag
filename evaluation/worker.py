"""Run exactly one query in one engine. Imported only by the evaluation harness.

The parent process enforces a wall-clock deadline by terminating the worker
subprocess. This does not guarantee cancellation of work already queued by the
Ollama server, so keep the sequential batch reasonably small.
"""
from __future__ import annotations

import argparse
import json
import traceback

PREFIX = "EVAL_RESULT_JSON="


def run(engine: str, question: str) -> dict:
    if engine == "basic":
        from src.generation import answer_question
        return answer_question(question)
    if engine == "agentic":
        from src.graph import ask_agentic
        return ask_agentic(question)
    raise ValueError(f"Unknown engine: {engine}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--engine", required=True, choices=["basic", "agentic"])
    parser.add_argument("--question", required=True)
    args = parser.parse_args()
    try:
        result = run(args.engine, args.question)
        payload = {"ok": True, "result": result}
    except Exception as e:
        payload = {
            "ok": False,
            "error_type": type(e).__name__,
            "error": str(e),
            "traceback_tail": traceback.format_exc(limit=2),
        }
    print(PREFIX + json.dumps(payload, ensure_ascii=False, default=str), flush=True)


if __name__ == "__main__":
    main()
