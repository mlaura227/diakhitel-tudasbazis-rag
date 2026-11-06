from __future__ import annotations
import copy
from pathlib import Path
import re
from src.config import resolve_source_path
from src.legal_reasoning import (
    plan_legal, missing_roles, answer_legal, targeted_legal_hits,
    TITLE_FRAGMENTS,
)
from src.query_policy import norm
from src.generation import answer_question
from src.graph import grade_retrieval


def regulation_hits():
    path = resolve_source_path('USZ_LAK_Bankszamla_20261015-knowledge.md')
    text = path.read_text(encoding='utf-8-sig')
    sections = []
    for part in text.split('\n## ')[1:]:
        title, body = part.split('\n', 1)
        if '*Forrás:' not in body:
            continue
        pages = re.search(r'\*Forrás:\s*(.*?)\s+oldal', body)
        section = {
            'content': 'Forrásdokumentum: Üzletszabályzat – Fizetési számlák (Bankszámlák)\n## ' + title + '\n' + body,
            'metadata': {
                'doc_id': 'account_rules_20261015', 'section_id': title,
                'section_title': title, 'document_title': 'Üzletszabályzat – Fizetési számlák (Bankszámlák)',
                'document_type': 'account_regulation', 'page': pages.group(1).strip(' .') if pages else 'unknown',
                'source_representation': 'interpreted_knowledge', 'effective_from': '2026-10-15',
            },
            'distance': 0.5,
        }
        sections.append(section)
    return sections


def selected(question):
    plan = plan_legal(question)
    assert plan is not None
    hits = [hit for hit in regulation_hits()
            if any(norm(fragment) in norm(hit['metadata']['section_title'])
                   for role, fragment in TITLE_FRAGMENTS.items() if role in plan.roles)]
    assert not missing_roles(plan, hits)
    return hits


def test_legal_plan_does_not_override_tariffs():
    assert plan_legal('Mennyibe kerül az OTP Diákhitel Számla számlavezetése?') is None
    assert plan_legal('Mennyibe kerül egy 300 000 Ft-os OTP ATM készpénzfelvétel?') is None


def test_legal_plan_contract_end_with_nonadjacent_terms():
    q = 'Mikor és milyen feltételekkel szűnik meg az OTP Diákhitel Számla szerződése?'
    assert plan_legal(q).roles == ('duration',)


def test_account_opening_conditions_complete_and_cited():
    q = 'Milyen feltételekkel nyitható OTP Diákhitel Számla?'
    r = answer_question(q, hits_override=selected(q))
    assert r['status'] == 'answered'
    assert r['mode'] == 'legal_evidence_assembly'
    for required in ('2027. június 30', '14. életév', 'VideóBank', 'Diákhitel Központ',
                     'egy időben egy Diákhitel', 'törvényes képviselő'):
        assert required in r['answer']
    assert {s['page'] for s in r['sources']} == {'33'}


def test_contract_end_and_base_account_afterwards():
    q = 'Mikor és milyen feltételekkel szűnik meg az OTP Diákhitel Számla szerződése?'
    r = answer_question(q, hits_override=selected(q))
    assert r['status'] == 'answered'
    assert 'folyósítási időszakának végétől számított egy évig' in r['answer']
    assert 'következő naptól' in r['answer']
    assert 'Bázis Számla' in r['answer']
    assert r['sources'][0]['page'] == '33–34'


def test_document_precedence_uses_its_own_clause():
    q = 'Melyik dokumentum az irányadó, ha a Diákhitel Hirdetmény és az általános Üzletszabályzat eltér?'
    r = answer_question(q, hits_override=selected(q))
    assert r['status'] == 'answered'
    assert 'hirdetmény/tájékoztató alkalmazandó' in r['answer']
    assert 'speciális rendelkezései megelőzik' in r['answer']
    assert r['sources'][0]['page'] == '6–7'


def test_multihop_opening_and_contract_expiry():
    q = 'Meddig lehet OTP Diákhitel Számlát nyitni, és mikor szűnik meg a szerződés?'
    r = answer_question(q, hits_override=selected(q))
    assert r['status'] == 'answered'
    assert len(r['sources']) == 3
    assert '2027. június 30' in r['answer']
    assert 'folyósítási időszakának végétől' in r['answer']


def test_all_quotes_are_exact_substrings():
    q = 'Milyen feltételekkel nyitható OTP Diákhitel Számla?'
    hits = selected(q)
    r = answer_question(q, hits_override=hits)
    mapping = {s['citation_id']: s for s in r['sources']}
    assert r['evidence_representation'] == 'interpreted_knowledge_not_verbatim_pdf'
    for ev in r['evidence']:
        meta = mapping[ev['source_id']]
        original = next(h for h in hits if h['metadata']['section_id'] == meta['section_id'])
        assert ev['quote'] in original['content']


def test_missing_clause_refused_not_filled_by_model():
    q = 'Milyen feltételekkel nyitható OTP Diákhitel Számla?'
    hits = [h for h in selected(q) if 'kiskorú' not in h['metadata']['section_title']]
    assert missing_roles(plan_legal(q), hits) == ('opening_channels',)
    r = answer_question(q, hits_override=hits)
    assert r['status'] == 'insufficient'
    assert r['mode'] == 'legal_missing_clauses'
    assert not r['evidence']


def test_grade_requires_all_relevant_legal_sections():
    q = 'Milyen feltételekkel nyitható OTP Diákhitel Számla?'
    hits = selected(q)
    assert grade_retrieval(q, hits)[0]
    assert not grade_retrieval(q, hits[:1])[0]


def test_targeted_retrieval_fills_missing_role_without_llm(monkeypatch):
    from src import legal_reasoning as lr
    q = 'Meddig lehet OTP Diákhitel Számlát nyitni, és mikor szűnik meg a szerződés?'
    relevant = selected(q)
    monkeypatch.setattr(lr, '_local_regulation_hits', lambda: tuple(regulation_hits()))
    recovered = targeted_legal_hits(plan_legal(q), relevant[:1])
    assert len(recovered) == 2
    assert missing_roles(plan_legal(q), relevant[:1] + recovered) == ()
    assert answer_legal(q, relevant[:1] + recovered)['status'] == 'answered'


def test_values_are_source_derived_not_hardcoded():
    q = 'Milyen feltételekkel nyitható OTP Diákhitel Számla?'
    hits = copy.deepcopy(selected(q))
    hits[0]['content'] = hits[0]['content'].replace('2027. június 30-áig', '2028. június 30-áig')
    r = answer_legal(q, hits)
    assert r['status'] == 'answered'
    assert '2028. június 30' in r['answer']
    assert '2027. június 30' not in r['answer']


def test_missing_clause_content_fails_closed():
    q = 'Mikor és milyen feltételekkel szűnik meg az OTP Diákhitel Számla szerződése?'
    hits = copy.deepcopy(selected(q))
    hits[0]['content'] = hits[0]['content'].replace('folyósítási időszakának végétől számított', 'egy ismeretlen időponttól')
    r = answer_legal(q, hits)
    assert r['status'] == 'insufficient'
    assert r['mode'] == 'legal_clause_validation_failed'


def test_post_expiry_fee_not_reused_from_zero_diakhitel_tariff():
    q = 'Mennyi a Diákhitel számlavezetési díja a szerződés megszűnése után?'
    plan = plan_legal(q)
    assert plan and plan.post_expiry_fee
    r = answer_question(q, hits_override=selected(q))
    assert r['status'] == 'insufficient'
    assert r['mode'] == 'legal_post_expiry_fee_not_documented'
    assert 'Bázis' in r['answer']
    assert '0 Ft' not in r['answer']
    assert r['evidence']


def test_langgraph_targeted_retrieval_integration_if_installed(monkeypatch):
    import pytest
    pytest.importorskip('langgraph')
    from src.graph import build_graph
    from src import legal_reasoning as lr
    q = 'Meddig lehet OTP Diákhitel Számlát nyitni, és mikor szűnik meg a szerződés?'
    all_hits = selected(q)
    monkeypatch.setattr(lr, '_local_regulation_hits', lambda: tuple(regulation_hits()))
    def limited_retriever(question, k=5, document_type=None):
        return all_hits[:1]
    graph = build_graph(
        retrieve_fn=limited_retriever,
        generate_fn=lambda question, hits_override: answer_question(question, hits_override=hits_override),
        rewrite_fn=lambda question: question,
    )
    state = graph.invoke({'question': q, 'search_query': q, 'retrieval_round': 0,
                          'rewrite_count': 0, 'steps': [], 'targeted_legal_done': False})
    assert state['result']['status'] == 'answered'
    assert state['retrieval_round'] == 2
    assert state['rewrite_count'] == 0
    assert any(t.startswith('targeted_legal:2') for t in state['steps'])
