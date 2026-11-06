"""
Small, deterministic decomposition and evidence-coverage gate.
This is NOT autonomous model planning. It only decomposes explicitly stated
independent subquestions and never rewrites monetary/geographic conditions.
The same answering and safety checks serve both Basic and Agentic RAG.
"""
from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Any
from src.query_policy import norm, plan_query
from src.applicability import check_applicability

Hit = dict[str, Any]

# Only split before an unambiguous new question expression, not after every occurrence of "és", where it may be part of a single legal condition.
_ASK = r'(?:mennyi(?:be)?|mikor|milyen|hogyan|hol|kell-e|lehet-e|van-e|ingyenes-e|mit\s+jelent|mi\s+a|mi\s+az)\b'
_SPLIT = re.compile(r'(?:[,;]\s*|\s+)\bés\s+(?=' + _ASK + r')', re.I)


@dataclass(frozen=True)
class FacetPlan:
    question: str
    facets: tuple[str, ...]


def plan_facets(question: str) -> FacetPlan | None:
    """Split only explicit multi-question expressions (max 4)."""
    source = question.strip()
    if not source:
        return None
    parts = [p.strip(' ,;?\n') for p in _SPLIT.split(source) if p.strip(' ,;?\n')]
    # Handle multiple full questions in one message.
    if len(parts) == 1 and source.count('?') >= 2:
        parts = [p.strip(' ,;?\n') for p in re.split(r'\?\s*', source) if p.strip(' ,;?\n')]
        # A follow-up like "Hogyan definiálja az OTP?" refers to the same concept, not a new information requirement. Do not force a second independently answerable facet out of an anaphoric follow-up.
        if len(parts) == 2 and not re.search(
            r'\b(?:sz[aá]mla|k[aá]rtya|utal[aá]s|beszed[eé]s|d[ií]j|atm|limit|'
            r'sepa|k[eé]szp[eé]nz|kivonat|hitel|deviza)\b', parts[1], re.I
        ):
            return None
    if not 2 <= len(parts) <= 4:
        return None
    if not all(len(p) >= 8 for p in parts):
        return None
    # Recover only *subject* context; do not add missing channel, place, year, or amount to any facet. This is necessary for follow-up elliptical clauses.
    original = norm(source)
    common = (
        'SEPA euróátutalás' if 'sepa' in original else
        'OTP Diákhitel Számla' if 'diakhitel' in original else
        'banki fizetési számla'
    )
    facets: list[str] = []
    for p in parts:
        folded = norm(p)
        if ('diakhitel' not in folded and 'sepa' not in folded and
            ('szamla' not in folded or common.startswith('SEPA'))):
            facets.append(f'{common}: {p}?')
        else:
            facets.append(p + '?')
    return FacetPlan(source, tuple(facets))


def _answer_one(facet: str, hits: list[Hit]) -> dict[str, Any] | None:
    """Only source-backed deterministic paths. No LLM call in the grade node."""
    from src.answer_router import answer_from_sources
    from src.extractive import extract_answer
    from src.legal_reasoning import plan_legal, answer_legal
    plan = plan_query(facet)
    if plan.scope != 'in_scope':
        return None
    result = answer_from_sources(facet, plan, hits)
    if result is None:
        legal_plan = plan_legal(facet)
        result = answer_legal(facet, hits, legal_plan) if legal_plan else extract_answer(facet, hits)
    return check_applicability(facet, result) if result else None


def missing_facets(plan: FacetPlan, hits: list[Hit]) -> tuple[str, ...]:
    return tuple(facet for facet in plan.facets
                 if (_answer_one(facet, hits) or {}).get('status') != 'answered')


def answer_facets(plan: FacetPlan, hits: list[Hit]) -> dict[str, Any]:
    """Join only validated literal quotations; name every unresolved facet."""
    from src.extractive import response
    from src.tariff_rules import section

    citations: list[tuple[Hit, str]] = []
    parts: list[str] = []
    missing: list[str] = []
    trace = []
    for facet in plan.facets:
        result = _answer_one(facet, hits)
        status = (result or {}).get('status', 'insufficient')
        trace.append({'facet': facet, 'status': status,
                      'mode': (result or {}).get('mode', 'missing_evidence')})
        if status != 'answered':
            missing.append(facet)
            continue
        # Evidence for every answered facet must be independently verifiable.
        refs = {s['citation_id']: s for s in result.get('sources', [])}
        supported: list[tuple[Hit, str]] = []
        for item in result.get('evidence', []):
            meta = refs.get(item.get('source_id'))
            if not meta:
                continue
            hit = section(hits, meta.get('doc_id'), meta.get('section_id'))
            quote = item.get('quote', '')
            if hit and quote and quote in hit.get('content', ''):
                supported.append((hit, quote))
        if not supported:
            missing.append(facet)
            trace[-1]['status'] = 'insufficient'
            continue
        citations.extend(supported)
        parts.append(result['answer'].strip())

    if not parts:
        return {'status': 'insufficient', 'answer':
            'A több részből álló kérdéshez nem található valamennyi szükséges, ellenőrzött bizonyíték.',
            'sources': [], 'evidence': [], 'mode': 'facet_coverage_missing',
            'missing_facets': missing, 'facet_trace': trace}
    body = ' '.join(parts)
    if missing:
        body += (' A következő kérdésrész(ek)re nem áll rendelkezésre ellenőrzött válasz: '
                 + ' | '.join(missing) + '.')
    output = response(body, citations)
    output['status'] = 'partial' if missing else 'answered'
    output['mode'] = 'facet_coverage_assembly'
    output['missing_facets'] = missing
    output['facet_trace'] = trace
    # Final audit uses the original full question as well.
    return check_applicability(plan.question, output)
