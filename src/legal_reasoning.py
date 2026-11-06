"""
Bounded, source-grounded reasoning for OTP Diákhitel contractual clauses.
This module intentionally does NOT calculate fees or apply general OTP product
rules to the Diákhitel account. Legal answers quote the *interpreted knowledge*
sections, not the PDF verbatim. Fee/definition routing remains unchanged.
"""
from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
import re
from typing import Any

from src.query_policy import norm

REG_DOC = 'account_rules_20261015'

# Distinct source sections, not fee tables. Title fragments tolerate hashed IDs.
TITLE_FRAGMENTS = {
    'opening_core': 'otp diakhitel szamla – termekcel, megnyitasi hatarido',
    'opening_channels': 'otp diakhitel szamla – kiskoru tulajdonos',
    'duration': 'otp diakhitel szamla – technikai bazis elnevezes',
    'precedence': 'szerzodeses iratok elsobbsege es termekspecifikus',
}

# These strings identify clauses in the source, NOT answer amounts or terms.
CLAUSE_KEYS = {
    'opening_core': (
        'otp-t jelolte szamlavezeto',
        'egy idoben',
        'jogosult az ugyfel uj otp diakhitel szamlat nyitni',
        '14. eletevet',
    ),
    'opening_channels': (
        'csak otp-bankfiokban vagy otp videobankon',
        'barmilyen jogviszonyban allt',
        'torvenyes kepviselo',
    ),
    'duration': (
        'folyositasi idoszakanak vegetol szamitott',
        'megszunes utan',
    ),
    'precedence': (
        'ertelmezesi sorrend',
        'ha az egyedi szerzodes',
        'specialis rendelkezesei megelozik',
    ),
}


@dataclass(frozen=True)
class LegalPlan:
    roles: tuple[str, ...]
    post_expiry_fee: bool = False


def plan_legal(question: str) -> LegalPlan | None:
    """Identify clauses required for a question, without generating an answer.

    Narrow to contractual questions: a fee question must still use the
    tariff-aware path; a general question about another OTP product is excluded
    by the upstream scope gate.
    """
    q = norm(question)
    fee_like = any(x in q for x in ('mennyibe', 'hany ft', 'mennyi a dija', 'dijazasa',
                                     'szamlavezetesi dij', 'szamlazaras dija', 'atutalas dija'))
    after_end = any(x in q for x in (
        'szerzodes lejarata utan', 'szerzodes megszunese utan',
        'szerzodes utan', 'diakhitel idoszak utan', 'diakhitel idoszak veget',
        'szerzodes lejarta utan', 'egy ev lejarta utan',
        'bazis szamlara valtas utan', 'bazis szamlava alakul',
    ))
    if fee_like:
        # After expiry, the 0 Ft Diákhitel tariff must NOT be carried forward.
        if after_end and ('diakhitel' in q or 'bazis szamla' in q):
            return LegalPlan(('duration',), post_expiry_fee=True)
        return None
    is_priority = (('hirdetmeny' in q or 'uzletszabalyzat' in q or 'egyedi szerzodes' in q)
                   and any(s in q for s in ('elter', 'elsobb', 'iranyad', 'sorrend', 'utkozik', 'felulir')))
    is_diakhitel = ('diakhitel' in q or ('otp' in q and 'szerzodes' in q))
    is_open = is_diakhitel and any(s in q for s in (
        'nyithat', 'megnyit', 'szamlanyitas', 'szamlat nyit', 'nyitas feltetel',
        'jogosult szamla',
    ))
    is_duration = is_diakhitel and any(s in q for s in (
        'szerzodes vege', 'szerzodes lejar', 'szerzodes megszun', 'szerzodese megszun',
        'szerzodes meddig', 'szerzodes idotartam', 'folyositasi idoszak',
        'mikor szunik', 'mikor er veget', 'meddig el', 'megszunes utan',
        'bazis szamlava', 'bazis szamlara valt',
    ))
    # Some prompts combine opening date and end of contract.
    if is_diakhitel and (('szunik meg' in q or 'megszunes' in q) and 'szerzod' in q):
        is_duration = True
    roles: list[str] = []
    if is_priority:
        roles.append('precedence')
    if is_open:
        roles.extend(('opening_core', 'opening_channels'))
    if is_duration:
        roles.append('duration')
    return LegalPlan(tuple(dict.fromkeys(roles))) if roles else None


def _role_of_hit(hit: dict[str, Any]) -> str | None:
    m = hit.get('metadata', {})
    if m.get('doc_id') != REG_DOC:
        return None
    title = norm(str(m.get('section_title', '')))
    return next((role for role, prefix in TITLE_FRAGMENTS.items()
                 if prefix in title), None)


def missing_roles(plan: LegalPlan, hits: list[dict[str, Any]]) -> tuple[str, ...]:
    present = {_role_of_hit(h) for h in hits}
    return tuple(role for role in plan.roles if role not in present)


@lru_cache(maxsize=1)
def _local_regulation_hits() -> tuple[dict[str, Any], ...]:
    """Read the *same indexed Markdown sections* only when a graph retry needs them.

    This is a targeted source lookup (not an LLM hallucination or another
    embedding query). It does not alter ChromaDB or source data.
    """
    from src.document_loader import load_documents
    docs = load_documents()
    return tuple({
        'content': doc.page_content,
        'metadata': dict(doc.metadata),
        'retrieval_method': 'targeted_legal_section',
        'distance': None,
    } for doc in docs if doc.metadata.get('doc_id') == REG_DOC)


def targeted_legal_hits(plan: LegalPlan, current_hits: list[dict[str, Any]]) -> list[dict[str, Any]]:
    wanted = set(missing_roles(plan, current_hits))
    if not wanted:
        return []
    selected = []
    for hit in _local_regulation_hits():
        if _role_of_hit(hit) in wanted:
            selected.append(hit)
            wanted.remove(_role_of_hit(hit))
        if not wanted:
            break
    return selected


def _sentences(hit: dict[str, Any]) -> list[str]:
    """Split only authored prose, keeping the exact substring for citation.

    Not a transcription parser: this is applied to the already interpreted
    knowledge sections, which have one prose paragraph and a source line.
    """
    lines = []
    for line in hit['content'].splitlines():
        if not line.strip() or line.startswith(('## ', 'Forrásdokumentum:', '*Forrás:')):
            continue
        lines.append(line.strip())
    paragraph = ' '.join(lines)
    return [x.strip() for x in re.split(r'(?<=[.!?])\s+(?=[A-ZÁÉÍÓÖŐÚÜŰ])', paragraph) if x.strip()]


def _matching_sentence(hit: dict[str, Any], key: str) -> str | None:
    for sentence in _sentences(hit):
        if key in norm(sentence):
            # Evidence must be an exact substring of its section text. This is
            # stricter than merely matching a paraphrase from model output.
            if sentence in hit['content']:
                return sentence
    return None


def _clean_for_answer(quote: str) -> str:
    # Content stays source-derived; remove only Markdown emphasis for UI prose.
    return quote.replace('**', '').strip()


def answer_legal(question: str, hits: list[dict[str, Any]], plan: LegalPlan | None = None) -> dict[str, Any] | None:
    plan = plan or plan_legal(question)
    if not plan:
        return None
    missing = missing_roles(plan, hits)
    if missing:
        return {
            'status': 'insufficient',
            'answer': 'A szerződéses kérdéshez szükséges összes üzletszabályzati szakasz nem áll rendelkezésre a visszakeresett bizonyítékokban.',
            'sources': [], 'evidence': [], 'attempts': 0,
            'mode': 'legal_missing_clauses', 'missing_legal_roles': list(missing),
        }
    pairs = []
    # Don't silently ignore missing applicability conditions inside a source.
    for role in plan.roles:
        source = next((h for h in hits if _role_of_hit(h) == role), None)
        if not source:
            return None
        for key in CLAUSE_KEYS[role]:
            matched = _matching_sentence(source, key)
            if matched is None:
                return {
                    'status': 'insufficient',
                    'answer': 'A szükséges jogi feltétel nem igazolható egyértelműen a betöltött szakasz szövegéből.',
                    'sources': [], 'evidence': [], 'attempts': 0,
                    'mode': 'legal_clause_validation_failed',
                }
            if not any(matched == old for _, old in pairs):
                pairs.append((source, matched))
    if not pairs:
        return None
    from src.extractive import response
    if plan.post_expiry_fee:
        # The contractual document establishes the transition, but does not
        # specify the Bázis tariff applicable at the user's future date.
        # Preserve an evidence-bearing refusal instead of reusing 0 Ft.
        duration_pair = next((item for item in pairs
                              if 'megszunes utan' in norm(item[1])), pairs[-1])
        result = response(
            'A Diákhitel-számlaszerződés megszűnése után a Bázis Számla '
            'feltételei és a hivatkozott Bázis/Junior díjhirdetmény az irányadó. '
            'A lejárat utáni pontos díj ebből a forrásrészletből nem állapítható meg.',
            [duration_pair],
        )
        result['status'] = 'insufficient'
        result['mode'] = 'legal_post_expiry_fee_not_documented'
        result['legal_roles'] = list(plan.roles)
        result['evidence_representation'] = 'interpreted_knowledge_not_verbatim_pdf'
        return result
    answer = ' '.join(_clean_for_answer(s) for _, s in pairs)
    result = response(answer, pairs)
    result['mode'] = 'legal_evidence_assembly'
    result['legal_roles'] = list(plan.roles)
    result['evidence_representation'] = 'interpreted_knowledge_not_verbatim_pdf'
    return result
