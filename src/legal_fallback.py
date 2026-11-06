"""
Conservative source-quotation fallback for unrecognized legal paraphrases.
Does not claim to interpret a complete contract, calculate a tariff or perform
semantic entailment. An authored knowledge sentence is cited verbatim and the
result is marked PARTIAL unless it directly negates or restricts a yes/no ask.
"""
from __future__ import annotations

import re
from typing import Any
from src.query_policy import norm
from src.extractive import response

_STOP = {
    'diakhitel', 'otp', 'szamla', 'szamlam', 'bank', 'banki', 'szamlara',
    'szamlahoz', 'szamlamban', 'lakossagi', 'vagyok', 'volna', 'ebben',
    'milyen', 'mikor', 'hogyan', 'miert', 'lehet', 'kerhetem', 'ugye',
    'biztosan', 'egyik', 'masik', 'vagy', 'akkor', 'kell', 'tudom',
    'azonban', 'ennek', 'ezzel', 'tovabb', 'most', 'ujabb',
}


def _terms(value: str) -> set[str]:
    return {w[:7] for w in re.findall(r'[a-z0-9]{5,}', norm(value))
            if w not in _STOP and not w.startswith(('diakhitel', 'szaml'))
            and not (w.startswith('bank') and not w.startswith('bankfiok'))}


def _sentences(hit: dict[str, Any]) -> list[str]:
    result = []
    for line in hit.get('content', '').splitlines():
        if not line.strip() or line.lstrip().startswith(('#', 'Forrásdokumentum:', '*Forrás:')):
            continue
        for sentence in re.split(r'(?<=[.!?])\s+(?=[A-ZÁÉÍÓÖŐÚÜŰ])', line):
            sentence = sentence.strip()
            if 35 <= len(sentence) <= 1200 and sentence in hit['content']:
                result.append(sentence)
    return result


def legal_quote_candidate(question: str, hits: list[dict[str, Any]]) -> tuple[dict[str, Any], str] | None:
    q = norm(question)
    # Only legal/service rules. Leave fees to condition-aware tariff extraction.
    if not any(w in q for w in (
        'szunetelt', 'nyit', 'igenyel', 'jogosult', 'kepvisel', 'szerzod',
        'mobilbank', 'videobank', 'bizalom ajanlat', 'megszun', 'hozzajarul',
        'szamlacsomag', 'vasarlas',
    )):
        return None
    q_terms = _terms(question)
    if not q_terms:
        return None
    ranked: list[tuple[int, dict[str, Any], str]] = []
    for hit in hits:
        meta = hit.get('metadata', {})
        if meta.get('document_type') != 'account_regulation':
            continue
        title = norm(str(meta.get('section_title', '')))
        # Do not borrow other OTP products' provisions for a Diákhitel query.
        is_dh = 'diakhitel' in q
        if is_dh and 'diakhitel' not in title:
            continue
        for sentence in _sentences(hit):
            s_terms = _terms(sentence)
            shared = q_terms.intersection(s_terms)
            if not shared:
                continue
            strong = len(shared)
            # Avoid treating the product name alone as sufficient support.
            if strong < 1:
                continue
            score = strong * 3
            if 'nem ' in norm(sentence) or 'kizarolag' in norm(sentence) or 'csak ' in norm(sentence):
                score += 1
            ranked.append((score, hit, sentence))
    if not ranked:
        return None
    ranked.sort(key=lambda item: item[0], reverse=True)
    best = ranked[0]
    # One content-word match is only safe as related information, not an answer.
    return best[1], best[2]


def answer_legal_fallback(question: str, hits: list[dict[str, Any]]) -> dict[str, Any] | None:
    found = legal_quote_candidate(question, hits)
    if not found:
        return None
    hit, quote = found
    result = response(
        'A betöltött OTP üzletszabályzat kapcsolódó rendelkezése: ' + quote.replace('**', ''),
        [(hit, quote)],
    )
    result['status'] = 'partial'
    result['mode'] = 'legal_quote_fallback'
    result['evidence_representation'] = 'interpreted_knowledge_not_verbatim_pdf'
    result['coverage_warning'] = ('A releváns szövegrészlet idézete nem bizonyítja önmagában, '
                                  'hogy a kérdés valamennyi feltétele teljesült.')
    return result
