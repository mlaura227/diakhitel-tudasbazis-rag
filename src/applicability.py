"""
Fail-closed applicability checks shared by Basic and Agentic RAG.
This checks only obvious mismatches. Passing is not a semantic entailment proof.
Never creates a tariff, changes a quoted value or assumes future fees.
"""
from __future__ import annotations

import re
from typing import Any
from src.query_policy import norm


def _cited_context(result: dict[str, Any]) -> str:
    """Only cited text/section titles count as the factual context."""
    names = ' '.join(str(s.get('section_title', '')) for s in result.get('sources', []))
    quotes = ' '.join(str(e.get('quote', '')) for e in result.get('evidence', []))
    return norm(names + ' ' + quotes)


def _fail(result: dict[str, Any], reason: str, issue: str) -> dict[str, Any]:
    failed = dict(result)
    failed['status'] = 'insufficient'
    failed['answer'] = reason
    failed['mode'] = 'applicability_guard'
    failed['applicability_issue'] = issue
    # Keep the recorded evidence as the *reason for exclusion*, not proof of a charge. It remains inspectable, but no tariff is asserted.
    return failed


def check_applicability(question: str, result: dict[str, Any]) -> dict[str, Any]:
    """Prevent unsupported fee transfer across geography, channel or time.

    Only checks *answered* or *partial* results with fee-like questions.
    A passing result still requires human grading for overall completeness.
    """
    if result.get('status') not in {'answered', 'partial'}:
        return result
    q = norm(question)
    if not any(x in q for x in (
        'dij', 'mennyibe', 'ingyen', 'koltseg', 'fizetek', 'fizetnem',
        'szazalek', 'dijmentes', 'kell fizet', '0 ft', 'draga', 'emel',
    )):
        return result
    context = _cited_context(result)
    if not context:
        return _fail(result,
            'A díjhoz nem áll rendelkezésre ellenőrizhető forráshivatkozás.',
            'missing_fee_evidence')

    # Distinguish where a card purchase happens; an account's geographical origin is NOT a replacement for a purchase's geographical attribute.
    if ('vasarl' in q or 'fizetek kartyaval' in q) and 'kulfold' in q:
        if ('vasarlas belfoldon' in context or 'belfoldi vasarlas' in context) and not any(
            x in context for x in ('vasarlas kulfoldon', 'kulfoldi vasarlas')
        ):
            return _fail(result,
                'A visszakeresett díjsor belföldi kártyás vásárlásra vonatkozik, ezért nem alkalmazható automatikusan külföldi vásárlásra.',
                'domestic_tariff_for_foreign_purchase')

    # A withdrawal in a *branch* must never use a tariff labelled ATM.
    branch = any(x in q for x in ('fiokban', 'bankfiokban', 'sajat fiok', 'fioknal'))
    withdrawal = ('keszpenzfelvet' in q or 'penzt veszek fel' in q or 'keszpenzt vennek fel')
    if branch and withdrawal and ('atm-bol' in context or 'atm-es' in context):
        if not any(x in context for x in ('sajat fiokban', 'idegen fiokban', 'bankfiokban')):
            return _fail(result,
                'A visszakeresett díjsor ATM-es készpénzfelvételről szól; nem igazolja a bankfióki készpénzfelvétel díját.',
                'atm_tariff_for_branch_withdrawal')

    # The printed document dates do not guarantee a future indexed fee. A different, explicit future-effective provision would need citing.
    future = bool(re.search(r'\b(?:202[7-9]|203\d)\b', q)) or any(
        x in q for x in ('jovore', 'kovetkezo evben', 'kovetkezo evre', 'inflacioval', 'indexal', 'majd mennyi lesz')
    )
    if future and ('mennyibe' in q or 'dij' in q or 'ingyen' in q or 'emel' in q or 'draga' in q):
        relevant_future_dates = re.findall(r'\b(?:202[7-9]|203\d)\b', q)
        # Require the *same* cited fee line to link a future year to an amount. A legal opening deadline somewhere else in the evidence is not evidence of a future fee schedule.
        quote_lines = [norm(str(e.get('quote', ''))) for e in result.get('evidence', [])]
        source_has_future_rate = bool(relevant_future_dates) and all(
            any(year in line and (re.search(r'\bft\b', line) or '%' in line)
                for line in quote_lines)
            for year in relevant_future_dates
        )
        answer_explicitly_qualifies = any(x in norm(str(result.get('answer', ''))) for x in (
            'nem allapithato meg', 'nem kovetkezik', 'nem garantal',
            'nem biztos', 'nem lehet megallapitani', 'jovobeli dij',
            'nem viheto tovabb', 'nem jelenti azt', 'nem ismert',
        ))
        if not source_has_future_rate and not answer_explicitly_qualifies:
            return _fail(result,
                'A forrásban szereplő jelenlegi díj nem bizonyítja a kérdezett jövőbeli időpontra érvényes díjat vagy indexálást.',
                'future_fee_not_established')

    return result
