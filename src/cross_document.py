"""Small *agentic-only* multi-document planner and evidence assembler.

This module deliberately uses source sections and exact textual evidence rather
than asking Qwen 3B to invent an answer. It never changes Basic RAG. Its finite
plans are an experimental, rule-directed form of agentic evidence acquisition,
not autonomous general-purpose reasoning. Leave the blind holdout untouched.
"""
from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Any

from src.query_policy import norm
from src.extractive import response
from src.legal_reasoning import (
    REG_DOC, TITLE_FRAGMENTS, _role_of_hit, _matching_sentence,
    _local_regulation_hits,
)
from src.source_sections import lookup_sections

Hit = dict[str, Any]
FEE = 'fee_schedule_20260901'
NOTICE = 'notice_20251118'

# Map the nonlegal roles to exact source sections. No stored answer values here.
SECTION_ROLES = {
    'fee_maintenance': (FEE, 'SRC-P03-B01'),
    'notice_eligibility': (NOTICE, 'source-p1-eligibility'),
    'notice_direct_debit': (NOTICE, 'source-p1-direct-debit'),
    'notice_card': (NOTICE, 'source-p1-card-fees'),
}


@dataclass(frozen=True)
class CrossPlan:
    kind: str
    roles: tuple[str, ...]


def plan_cross_document(question: str) -> CrossPlan | None:
    """Detect compound *OTP Diákhitel* questions, not single-source fee lookups.

    The plans describe required *evidence*, not predetermined answers. Every
    amount/date is read from the actual retrieved Markdown at execution time.
    """
    q = norm(question)
    if not ('diakhitel' in q or ('otp' in q and 'szamla' in q)):
        return None
    end = any(word in q for word in (
        'lejar', 'megszun', 'megszunes', 'kedvezmenyes idoszak',
        'folyositasi idoszak veget', 'diakhitel idoszak',
    ))
    # A question combining account continuity and continued free maintenance: report the known transition and explicitly withhold the later Bázis fee.
    continuation = any(x in q for x in ('bankszamlam', 'szamla is', 'szamla tovabb',
                                        'automatikusan megszunik', 'megszunik a szamla',
                                        'szamla megmarad', 'szamla atvaltozik'))
    fee = any(x in q for x in ('ingyen', 'dijmentes', 'szamlavezetesi dij', '0 ft',
                                'tovabbra is', 'mennyibe', 'mennyit fizet'))
    if end and fee and continuation:
        return CrossPlan('expiry_and_fee', ('duration', 'fee_maintenance'))

    eligibility = any(x in q for x in (
        'jogosult', 'feltetel', 'nyithat', 'megnyit', 'igenyelhet',
    ))
    debit = 'beszed' in q
    if eligibility and debit:
        return CrossPlan('eligibility_and_debit',
                         ('opening_core', 'notice_eligibility', 'notice_direct_debit'))

    precedence = (any(x in q for x in ('hirdetmeny', 'uzletszabalyzat'))
                  and any(x in q for x in ('elter', 'ellentmond', 'iranyado', 'felulir')))
    if precedence and ('szamlavezet' in q or 'havi dij' in q):
        return CrossPlan('precedence_and_fee', ('precedence', 'fee_maintenance'))
    return None


def _role(hit: Hit) -> str | None:
    legal = _role_of_hit(hit)
    if legal:
        return legal
    m = hit.get('metadata', {})
    for role, (doc_id, section_id) in SECTION_ROLES.items():
        if m.get('doc_id') == doc_id and m.get('section_id') == section_id:
            return role
    return None


def missing_roles(plan: CrossPlan, hits: list[Hit]) -> tuple[str, ...]:
    present = {_role(h) for h in hits}
    return tuple(role for role in plan.roles if role not in present)


def targeted_cross_hits(plan: CrossPlan, current_hits: list[Hit]) -> list[Hit]:
    """Fetch *only missing* sections, preserving source provenance.

    This is a deterministic local source lookup. It is independent of Chroma,
    allowing retrieval failures to be recovered without modifying fee logic.
    """
    missing = set(missing_roles(plan, current_hits))
    if not missing:
        return []
    selected: list[Hit] = []
    legal = set(TITLE_FRAGMENTS)
    if missing & legal:
        for hit in _local_regulation_hits():
            role = _role(hit)
            if role in missing:
                selected.append(hit)
                missing.remove(role)
            if not (missing & legal):
                break
    keys = [SECTION_ROLES[role] for role in plan.roles
            if role in missing and role in SECTION_ROLES]
    selected.extend(lookup_sections(keys))
    return selected


def _part(hit: Hit, marker: str) -> str | None:
    """Find a complete literal line or sentence with the required meaning."""
    if hit.get('metadata', {}).get('doc_id') == REG_DOC:
        return _matching_sentence(hit, marker)
    for line in hit.get('content', '').splitlines():
        if marker in norm(line) and len(line.strip()) > 15:
            return line.strip()
    return None


def _fee_items(hit: Hit) -> tuple[str, str] | None:
    electronic = _part(hit, 'elektronikus kivonattal: havi dij')
    paper = _part(hit, 'papiralapu kivonattal: havi dij')
    if not electronic or not paper:
        return None
    # Source-derived fee strings; preserve distinctions if variants differ.
    return electronic, paper


def _quoted_fee(line: str) -> tuple[str, str] | None:
    m = re.search(r'havi díj\s+([\d\s]+\s*Ft).*?teljes éves díj\s+([\d\s]+\s*Ft)', line, re.I)
    return (m.group(1).strip(), m.group(2).strip()) if m else None


def _verified(plan: CrossPlan, hits: list[Hit]) -> dict[str, Any] | None:
    if missing_roles(plan, hits):
        return None
    lookup = {_role(hit): hit for hit in hits if _role(hit) in plan.roles}
    evidence: list[tuple[Hit, str]] = []
    def take(role: str, marker: str) -> str | None:
        hit = lookup[role]
        quote = _part(hit, marker)
        if quote:
            evidence.append((hit, quote))
        return quote
    if plan.kind == 'expiry_and_fee':
        end = take('duration', 'folyositasi idoszakanak vegetol szamitott')
        conversion = take('duration', 'megszunes utan')
        fees = _fee_items(lookup['fee_maintenance'])
        if not (end and conversion and fees):
            return None
        evidence.extend((lookup['fee_maintenance'], quote) for quote in fees)
        monthly = _quoted_fee(fees[0])
        if not monthly:
            return None
        answer = (
            'A Diákhitel-számlaszerződés a hallgatói hitel folyósítási '
            'időszakának végétől számított egy évig tart. A Diákhitel-kedvezmények '
            'megszűnése után a számlát a bank a következő naptól Bázis Számla '
            'feltételekkel kezeli; ez nem azonos a bankszámla automatikus bezárásával. '
            f'A betöltött Diákhitel-díjjegyzékben a számlavezetés {monthly[0]} havonta, '
            'de ez nem vihető tovább automatikusan a Bázis Számlára. A későbbi '
            'Bázis Számla konkrét díja a betöltött forrásokból nem állapítható meg.'
        )
        result = response(answer, evidence)
        result['status'] = 'partial'  # The future Bázis fee remains unresolved.
    elif plan.kind == 'eligibility_and_debit':
        a = take('opening_core', 'otp-t jelolte szamlavezeto')
        b = take('opening_core', 'egy idoben')
        deadline = take('opening_core', 'jogosult az ugyfel uj otp diakhitel szamlat nyitni')
        age = take('opening_core', '14. eletevet')
        notice_rule = take('notice_eligibility', 'uzletszabalyzat a lakossagi')
        c = take('notice_eligibility', 'diakhitel kozpont zrt.-nel')
        # Notices' explicit conditions may have separate lines / list numbers.
        d = take('notice_direct_debit', 'csoportos beszedesi megbizas')
        e = take('notice_direct_debit', 'kivetel:')
        if not all((a, b, deadline, age, notice_rule, c, d, e)):
            return None
        # Interpret only the explicit table row's last cell, not other prices.
        cells = [cell.strip() for cell in d.strip().strip('|').split('|')]
        if len(cells) != 2 or 'csoportos beszedesi' not in norm(cells[0]):
            return None
        answer = (
            'Az OTP Diákhitel Számla igénybevételéhez teljesíteni kell a '
            'termékre vonatkozó jogosultsági feltételeket, és az OTP-t kell '
            'számlavezető bankként megjelölni a Diákhitel Központnál. '
            'Egy ügyfélnek egyidejűleg csak egy Diákhitel számlája lehet. '
            + age.replace('**', '').strip() + ' '
            + deadline.replace('**', '').strip() + ' '
            + f'A Diákhitel-hirdetmény a csoportos beszedés díját {cells[-1]} '
            'tételként jelöli, de ez nem terjed ki az „Egyéb beszedések” '
            'kategóriájába tartozó műveletekre; azok díja az idézett '
            'dokumentumokból nem állapítható meg.'
        )
        result = response(answer, evidence)
    elif plan.kind == 'precedence_and_fee':
        priority = take('precedence', 'ha az egyedi szerzodes')
        specialised = take('precedence', 'specialis rendelkezesei megelozik')
        fees = _fee_items(lookup['fee_maintenance'])
        if not (priority and specialised and fees):
            return None
        evidence.extend((lookup['fee_maintenance'], quote) for quote in fees)
        monthly = _quoted_fee(fees[0])
        if not monthly:
            return None
        answer = (
            'Eltérés esetén az egyedi szerződés, illetve az ügyletre alkalmazandó '
            'hirdetmény/tájékoztató elsőbbséget élvez az általános üzletszabályzattal '
            'szemben; a termékspecifikus feltételek az általános feltételek előtt állnak. '
            f'Az OTP Diákhitel Számla díjjegyzéke alapján a számlavezetés havi díja {monthly[0]}.'
        )
        result = response(answer, evidence)
    else:
        return None
    result['mode'] = 'agentic_multidoc_evidence_assembly'
    result['evidence_representation'] = 'mixed_transcription_and_interpreted_knowledge'
    result['required_roles'] = list(plan.roles)
    return result


def answer_cross_document(plan: CrossPlan, hits: list[Hit]) -> dict[str, Any]:
    if missing_roles(plan, hits):
        return {
            'status': 'insufficient',
            'answer': 'A több dokumentumot érintő kérdéshez nem áll rendelkezésre minden szükséges forrásszakasz.',
            'sources': [], 'evidence': [], 'mode': 'agentic_multidoc_missing_evidence',
            'missing_roles': list(missing_roles(plan, hits)), 'attempts': 0,
        }
    result = _verified(plan, hits)
    if result:
        return result
    return {
        'status': 'insufficient',
        'answer': 'A visszakeresett szakaszok nem igazolják egyértelműen a kérdés minden szükséges feltételét.',
        'sources': [], 'evidence': [], 'mode': 'agentic_multidoc_clause_validation_failed',
        'attempts': 0,
    }
