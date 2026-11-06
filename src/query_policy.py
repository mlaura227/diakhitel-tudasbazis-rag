"""
Cheap, deterministic pre-retrieval scope and intent routing.
This is a *corpus boundary*, not a financial advice classifier. Explicitly
unsupported institutions/products are rejected before model or vector calls.
"""
from __future__ import annotations

from dataclasses import dataclass
import re
import unicodedata


def norm(s: str) -> str:
    return ''.join(c for c in unicodedata.normalize('NFKD', s.casefold()) if not unicodedata.combining(c))


@dataclass(frozen=True)
class QueryPlan:
    scope: str  # in_scope, other_bank, other_product, out_of_scope
    intents: tuple[str, ...]  # fee, definition, eligibility, document, general
    sections: tuple[tuple[str, str], ...]
    reason: str = ''


OTHER_BANK_RE = re.compile(r'\b(?:k\s*&\s*h|erste|mbh|revolut|raiffeisen|unicredit|cib|granit|magnet)[a-z]*\b', re.I)
OTHER_PRODUCTS_RE = re.compile(r'\b(?:junior|smart|premium|privat)\s+(?:szamla|szamlacsomag)', re.I)
DOMAIN_RE = re.compile(
    r'szaml|bank|dij|koltseg|atm|keszpenz|utal|sepa|deviza|forint|eur|kartya|'
    r'kivonat|igazolas|beszed|biztositas|hitel|kam(at|at)|sms|limit|otp|qvik|'
    r'folyoszaml|befizet|jovair|kedvezmeny|mobilbank|internetbank'
)
SERVICE_RE = re.compile(
    r'szaml|dij|koltseg|atm|keszpenz|utal|sepa|deviza|forint|eur|kartya|'
    r'kivonat|igazolas|beszed|biztositas|hitel|kamat|sms|limit|qvik|'
    r'befizet|jovair|kedvezmeny|mobilbank|internetbank'
)


def plan_query(question: str) -> QueryPlan:
    q = norm(question)
    if OTHER_BANK_RE.search(q):
        # The bank *providing the account* is outside the corpus. If the question explicitly concerns an OTP card at another bank's ATM, bank name may simply identify the ATM network, not the product.
        if not (('otp' in q or 'diakhitel' in q) and ('atm' in q or 'automata' in q)
                and not any(w in q for w in ('ugyfele', 'ugyfel', 'szamlaja', 'szamlajan'))):
            return QueryPlan('other_bank', (), (), 'A megnevezett bank termékdíjai nem szerepelnek a betöltött OTP-dokumentumokban.')
        if 'vagy' in q or 'osszehasonlit' in q or 'melyik olcsobb' in q:
            return QueryPlan('other_bank', (), (), 'Más bank díjaival történő összehasonlításhoz nincs elegendő adat.')
    if OTHER_PRODUCTS_RE.search(q) and 'diakhitel' not in q:
        return QueryPlan('other_product', (), (), 'A betöltött kondíciók kizárólag az OTP Diákhitel Számlára vonatkoznak.')
    if not SERVICE_RE.search(q):
        return QueryPlan('out_of_scope', (), (), 'A kérdés nem a betöltött OTP Diákhitel Számla dokumentumaira vonatkozik.')
    # "Mit jelent az OTP?" is an acronym question, but not a supported banking-service definition. Ignore bare OTP as a sufficient domain cue.
    if re.search(r'\bmit jelent (?:az )?otp\b', q) and not SERVICE_RE.search(q.replace('otp', '')):
        return QueryPlan('out_of_scope', (), (), 'Az OTP rövidítés eredetét a betöltött dokumentumok nem definiálják.')

    definition = any(t in q for t in ('mit jelent', 'mi az a ', 'mi az ', 'milyen muvelet', 'fogalm', 'definici', 'definial', 'mit neveznek', 'mire utal'))
    fee = any(t in q for t in ('dij', 'koltseg', 'mennyibe', 'fizetek', 'fizetnem', 'fizetni', 'ingyen', 'dijmentes', 'szamolnak fel', 'szamitanak fel', 'havi', 'eves', '0 ft', 'utalasnak van', 'kulonbozik'))
    # Do not interpret "mi a ... díja" as pure definition.
    if 'hataly' in q and ('2026' in q or 'dijjegyzek' in q):
        intents = ('document',)
    elif fee and definition:
        intents = ('definition', 'fee')
    elif fee:
        intents = ('fee',)
    elif definition:
        intents = ('definition',)
    elif any(t in q for t in ('jogosult', 'feltetel', 'igenyelhet', 'kedvezmeny')):
        intents = ('eligibility',)
    elif any(t in q for t in ('hataly', 'datum', '2026', '2025')):
        intents = ('document',)
    else:
        intents = ('general',)

    keys: list[tuple[str,str]] = []
    def add(doc: str, section: str) -> None:
        item = (doc, section)
        if item not in keys: keys.append(item)
    fee_doc, notice, gloss = 'fee_schedule_20260901', 'notice_20251118', 'glossary_undated'

    if 'szamlavezet' in q or 'szamladij' in q:
        add(fee_doc, 'SRC-P03-B01')
    if 'mobilbank' in q or 'internetbank' in q:
        add(fee_doc, 'SRC-P03-B03')
    if any(x in q for x in ('szamlazar', 'megszuntetem', 'szamlavalt')):
        add(fee_doc, 'SRC-P03-B02')
    if 'kivonat' in q:
        if any(x in q for x in ('masolat','utolag')): add(fee_doc, 'SRC-P09-B03')
        if any(x in q for x in ('papir', 'havi', 'rendszeres', 'postaz')): add(fee_doc, 'SRC-P03-B01')
    if any(x in q for x in ('fedezetigazolas','egyenlegigazolas','igazolas kiallit')):
        add(fee_doc, 'SRC-P09-B03')
    if 'sms' in q or 'ertesito' in q:
        add(fee_doc, 'SRC-P04-B01')
        add(fee_doc, 'SRC-P04-B02')
        if 'csomag' in q or 'havi' in q or 'eves' in q: add(fee_doc, 'SRC-P04-B03')
    if any(x in q for x in ('sepa', 'euroat')):
        add(fee_doc, 'SRC-P06-B01')
        if 'definition' in intents: add(gloss, 'section-009')
    if 'utal' in q or 'atutal' in q:
        if 'rendszeres' in q: add(fee_doc, 'SRC-P05-B02')
        if any(x in q for x in ('forint','fiok', 'papir', 'eseti')): add(fee_doc, 'SRC-P05-B01')
    if 'csoportos' in q or 'beszedes dij' in q: add(notice, 'source-p1-direct-debit')
    if 'devizajovair' in q: add(fee_doc,'SRC-P06-B03')
    if 'keszpenzfelvet' in q or (('atm' in q or 'automata' in q) and any(x in q for x in ('felv','felven', 'veszek fel', 'venni', 'kivesz'))):
        if any(x in q for x in ('kulfold', 'nemetorszag', 'egt', 'eurot fel', 'ausztria')):
            add(fee_doc, 'SRC-P08-B01')
        else:
            add(fee_doc, 'SRC-P07-B04')
    if 'befizet' in q and ('atm' in q or 'keszpenz' in q): add(fee_doc, 'SRC-P08-B03')
    if 'kartya' in q:
        if any(x in q for x in ('potlas', 'potkartya', 'potol', 'kibocsat', 'eves dij','elso eves','masodik eves')):
            add(notice, 'source-p1-card-fees')
            add(fee_doc, 'SRC-P06-B04')
        if 'vasarlas' in q or 'vasarlo' in q: add(fee_doc, 'SRC-P07-B02')
        if 'hitelkartya' in q: add(fee_doc, 'SRC-P07-B01')
        if 'limit' in q: add(fee_doc, 'SRC-P09-B02')
    if 'limitmodositas' in q or ('atm' in q and 'limit' in q): add(fee_doc,'SRC-P09-B02')
    if 'folyoszamlahitel' in q or 'hitelkeret' in q: add(fee_doc,'SRC-P09-B01')
    if 'biztositas' in q or 'kedvezmeny' in q: add(notice, 'source-p1-additional-benefits')
    if 'jogosult' in q or 'kedvezmenyes' in q: add(notice, 'source-p1-eligibility')
    if 'hataly' in q:
        add(notice, 'source-document-metadata')
        add(fee_doc, 'section-001')

    if 'definition' in intents:
        # A glossary is always the primary authority for definitions.
        definitions = {
            'sepa': 'section-009', 'beszed': 'section-008', 'folyoszamlahitel':'section-019',
            'devizajovair': 'section-011', 'keszpenzbefizet': 'section-018',
            'limitmodositas': 'section-020', 'betetikartya':'section-012',
        }
        for term, sid in definitions.items():
            if term in q: add(gloss, sid)
    return QueryPlan('in_scope', intents, tuple(keys))


def scope_result(plan: QueryPlan) -> dict:
    return {
        'status': 'insufficient',
        'answer': plan.reason + ' A rendszer csak az OTP Diákhitel Számla betöltött dokumentumaiból válaszol.',
        'sources': [], 'evidence': [], 'mode': 'scope_gate', 'retrieval_trace': [], 'attempts': 0,
    }
