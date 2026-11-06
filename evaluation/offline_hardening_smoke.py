"""Offline status smoke check using structured Markdown lookup only.

NOT an end-to-end Chroma, BGE-M3, Qwen or LangGraph evaluation. This reuses
known development questions and checks expected status, not factual entailment.
"""
from __future__ import annotations
import json
from collections import Counter
from pathlib import Path

from src.generation import answer_question
from src.query_policy import plan_query
from src.source_sections import lookup_sections

DATASET = Path(__file__).resolve().parent / 'test_questions.jsonl'


def run() -> int:
    counts=Counter()
    wrong=[]
    for line in DATASET.read_text(encoding='utf-8-sig').splitlines():
        if not line.strip(): continue
        case=json.loads(line)
        question=case['question']
        plan=plan_query(question)
        hits=lookup_sections(plan.sections) if plan.scope=='in_scope' else []
        result=answer_question(question,hits_override=hits)
        ok=result['status']==case['expected_status']
        counts[case['category']]+=1
        counts[(case['category'],'matched')]+=int(ok)
        if not ok:
            wrong.append((case['id'], case['expected_status'],result['status']))
    for cat in ('fee_lookup','multi_section','glossary','unsupported','false_premise','other_bank','out_of_scope'):
        print(f"{cat}: {counts[(cat,'matched')]}/{counts[cat]} status matches")
    total=sum(counts[cat] for cat in ('fee_lookup','multi_section','glossary','unsupported','false_premise','other_bank','out_of_scope'))
    matches=sum(counts[(cat,'matched')] for cat in ('fee_lookup','multi_section','glossary','unsupported','false_premise','other_bank','out_of_scope'))
    print(f'OFFLINE KNOWN-QUESTION status matches: {matches}/{total}')
    print('This is NOT answer accuracy, latency, or the live 44-case evaluation.')
    if wrong: print('Status mismatches:',wrong)
    return 0 if not wrong else 1


if __name__=='__main__':
    raise SystemExit(run())
