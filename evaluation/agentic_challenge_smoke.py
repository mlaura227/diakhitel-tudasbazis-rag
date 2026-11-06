"""
Small visible development probe. NOT an independent holdout evaluation.
Run with `python -m evaluation.agentic_challenge_smoke` after installing the patch.
Requires working local Chroma/Ollama and project dependencies.
"""
from __future__ import annotations

import time
from src.generation import answer_question
from src.graph import ask_agentic

QUESTIONS = [
    (
        'Contract expiry + current vs later fee',
        'Ha lejár az OTP Diákhitel Számlám kedvezményes időszaka, '
        'automatikusan megszűnik a bankszámlám is, és továbbra is ingyenes marad a számlavezetés?',
    ),
    (
        'Eligibility + group direct debit exception',
        'Ha jogosult vagyok OTP Diákhitel Számlára, a csoportos beszedés '
        'díjmentes, vagy az Egyéb beszedésekre is igaz?',
    ),
    (
        'Document precedence + maintenance fee',
        'Ha az OTP Diákhitel Hirdetmény és az Üzletszabályzat eltér, '
        'melyik az irányadó, és mennyi a havi számlavezetési díj?',
    ),
]


def main() -> None:
    print('Development probe only — do not report these as unseen evaluation results.\n')
    for title, question in QUESTIONS:
        print('=' * 78)
        print(title + '\n' + question)
        for engine, function in [('basic', answer_question), ('agentic', ask_agentic)]:
            start = time.perf_counter()
            try:
                result = function(question)
            except Exception as exc:
                print(f'[{engine}] ERROR: {type(exc).__name__}: {exc}')
                continue
            duration = time.perf_counter() - start
            sources = [f"{s.get('doc_id')}:{s.get('section_id')}" for s in result.get('sources', [])]
            print(f'[{engine}] status={result.get("status")} '
                  f'mode={result.get("mode")} latency={duration:.2f}s '
                  f'rounds={result.get("retrieval_rounds", "n/a")} '
                  f'rewrites={result.get("rewrite_count", "n/a")}')
            print('  answer:', result.get('answer', ''))
            print('  citations:', '; '.join(sources) or '(none)')
            if engine == 'agentic':
                print('  trace:', ' -> '.join(result.get('graph_trace', [])))
        print()


if __name__ == '__main__':
    main()
