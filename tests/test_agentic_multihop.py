from __future__ import annotations

from pathlib import Path
import copy

import pytest

from src.config import resolve_source_path
from src.query_policy import norm, plan_query
from src.source_sections import lookup_sections
from src.legal_reasoning import TITLE_FRAGMENTS
from src.cross_document import (
    plan_cross_document, missing_roles, answer_cross_document,
    targeted_cross_hits, SECTION_ROLES,
)
from src.graph import grade_retrieval, conflicting_section_payloads

Q_EXPIRY = (
    'Ha lejár az OTP Diákhitel Számlám kedvezményes időszaka, '
    'automatikusan megszűnik a bankszámlám is, és továbbra is '
    'ingyenes marad a számlavezetés?'
)
Q_DEBIT = (
    'Ha jogosult vagyok OTP Diákhitel Számlára, a csoportos beszedés '
    'díjmentes, vagy az Egyéb beszedésekre is igaz?'
)
Q_PRECEDENCE = (
    'Ha az OTP Diákhitel Hirdetmény és az Üzletszabályzat eltér, '
    'melyik az irányadó, és mennyi a havi számlavezetési díj?'
)


def regulation_sections():
    """Read exact authored Markdown sections, independently of Chroma/Ollama."""
    text = resolve_source_path('USZ_LAK_Bankszamla_20261015-knowledge.md').read_text(encoding='utf-8-sig')
    sections = {}
    for part in text.split('\n## ')[1:]:
        title, body = part.split('\n', 1)
        for role, fragment in TITLE_FRAGMENTS.items():
            if norm(fragment) in norm(title):
                sections[role] = {
                    'metadata': {
                        'doc_id': 'account_rules_20261015',
                        'section_id': 'TEST-' + role,
                        'document_title': 'Üzletszabályzat – Fizetési számlák (Bankszámlák)',
                        'document_type': 'account_regulation',
                        'section_title': title,
                        'page': '33–34' if role == 'duration' else '33',
                        'effective_from': '2026-10-15',
                    },
                    'content': 'Forrásdokumentum: Üzletszabályzat – Fizetési számlák (Bankszámlák)\n## ' + title + '\n' + body,
                    'retrieval_method': 'injected',
                }
    return sections


def needed_hits(question):
    plan = plan_cross_document(question)
    assert plan, question
    reg = regulation_sections()
    out = []
    for role in plan.roles:
        if role in reg:
            out.append(copy.deepcopy(reg[role]))
        else:
            fetched = lookup_sections([SECTION_ROLES[role]])
            assert len(fetched) == 1, role
            out.append(copy.deepcopy(fetched[0]))
    return out


@pytest.mark.parametrize('question,kind', [
    (Q_EXPIRY, 'expiry_and_fee'),
    (Q_DEBIT, 'eligibility_and_debit'),
    (Q_PRECEDENCE, 'precedence_and_fee'),
])
def test_plan_and_answer_all_compound_roles(question, kind):
    plan = plan_cross_document(question)
    assert plan.kind == kind
    hits = needed_hits(question)
    assert not missing_roles(plan, hits)
    assert grade_retrieval(question, hits)[0]
    result = answer_cross_document(plan, hits)
    assert result['status'] in ('answered', 'partial')
    assert result['mode'] == 'agentic_multidoc_evidence_assembly'
    assert len(result['sources']) >= 2
    references = {s['citation_id']: s for s in result['sources']}
    for e in result['evidence']:
        key = (references[e['source_id']]['doc_id'], references[e['source_id']]['section_id'])
        source = next(h for h in hits if (h['metadata']['doc_id'], h['metadata']['section_id']) == key)
        assert e['quote'] in source['content']


def test_expiry_does_not_extend_free_fee():
    result = answer_cross_document(plan_cross_document(Q_EXPIRY), needed_hits(Q_EXPIRY))
    assert result['status'] == 'partial'
    assert 'Bázis Számla' in result['answer']
    assert '0 Ft' in result['answer']  # earlier Diákhitel condition is allowed
    assert 'későbbi Bázis Számla konkrét díja' in result['answer']
    assert 'automatikus bezárásával' in result['answer']


def test_direct_debit_exception_is_not_dropped():
    result = answer_cross_document(plan_cross_document(Q_DEBIT), needed_hits(Q_DEBIT))
    assert result['status'] == 'answered'
    assert 'Egyéb beszedések' in result['answer']
    assert 'díjmentes' in result['answer']
    assert any('Kivétel:' in e['quote'] for e in result['evidence'])


def test_fee_is_from_document_not_an_unchanging_constant():
    plan = plan_cross_document(Q_PRECEDENCE)
    hits = needed_hits(Q_PRECEDENCE)
    fee_hit = next(h for h in hits if h['metadata']['doc_id'] == 'fee_schedule_20260901')
    fee_hit['content'] = fee_hit['content'].replace('havi díj 0 Ft;', 'havi díj 90 Ft;')
    result = answer_cross_document(plan, hits)
    assert result['status'] == 'answered'
    assert '90 Ft' in result['answer']
    assert not __import__('re').search(r'(?<!\d)0\s*Ft', result['answer'])


def test_missing_role_fails_closed_without_synthesizing_evidence():
    plan = plan_cross_document(Q_DEBIT)
    hits = needed_hits(Q_DEBIT)[:-1]
    assert missing_roles(plan, hits) == ('notice_direct_debit',)
    assert not grade_retrieval(Q_DEBIT, hits)[0]
    result = answer_cross_document(plan, hits)
    assert result['status'] == 'insufficient'
    assert result['mode'] == 'agentic_multidoc_missing_evidence'
    assert not result['sources']


def test_missing_exception_clause_fails_closed():
    plan = plan_cross_document(Q_DEBIT)
    hits = needed_hits(Q_DEBIT)
    debit = next(h for h in hits if h['metadata']['section_id'] == 'source-p1-direct-debit')
    debit['content'] = '\n'.join(line for line in debit['content'].splitlines() if 'Kivétel:' not in line)
    result = answer_cross_document(plan, hits)
    assert result['status'] == 'insufficient'
    assert result['mode'] == 'agentic_multidoc_clause_validation_failed'


def test_different_payloads_for_same_section_are_rejected():
    one = needed_hits(Q_PRECEDENCE)[-1]
    other = copy.deepcopy(one)
    other['content'] = other['content'].replace('0 Ft', '900 Ft')
    assert conflicting_section_payloads([one, other])
    other['retrieval_method'] = 'vector'
    assert not conflicting_section_payloads([one, other])


def test_out_of_scope_does_not_create_compound_plan():
    assert plan_cross_document('Mennyi a K&H diákhitel számla havi díja?') is None or plan_query('Mennyi a K&H diákhitel számla havi díja?').scope != 'in_scope'
    assert plan_query('szia mit jelent az otp').scope == 'out_of_scope'


def test_agentic_multihop_graph_recovers_missing_section_when_langgraph_is_present():
    pytest.importorskip('langgraph')
    from src.graph import build_graph
    hits = needed_hits(Q_DEBIT)
    initial = hits[:1]
    calls = []
    generated = []

    def retriever(question, **kwargs):
        calls.append(question)
        return initial

    def targeted(plan, current_hits):
        assert plan.kind == 'eligibility_and_debit'
        return [h for h in hits if h['metadata']['section_id'] != initial[0]['metadata']['section_id']]

    def generator(question, *, hits_override):
        from src.generation import answer_question
        generated.append(question)
        return answer_question(question, hits_override=hits_override)

    graph = build_graph(retrieve_fn=retriever, generate_fn=generator,
                        rewrite_fn=lambda q: q, targeted_fn=targeted)
    state = graph.invoke({'question': Q_DEBIT, 'search_query': Q_DEBIT,
                          'retrieval_round': 0, 'rewrite_count': 0,
                          'steps': [], 'targeted_legal_done': False},
                         config={'recursion_limit': 12})
    assert state['result']['status'] == 'answered'
    assert state['retrieval_round'] == 2  # 1 vector + 1 targeted source lookup
    assert state['rewrite_count'] == 0
    assert generated == [Q_DEBIT]
    assert any(step == 'targeted_multidoc:2' for step in state['steps'])
    assert state['steps'].count('grade:weak') == 1
    assert state['steps'].count('grade:sufficient') == 1
    assert calls == [Q_DEBIT]


def test_agentic_graph_rejects_conflicting_same_section_when_langgraph_is_present():
    pytest.importorskip('langgraph')
    from src.graph import build_graph
    hit = needed_hits(Q_PRECEDENCE)[-1]
    conflict = copy.deepcopy(hit)
    conflict['content'] = conflict['content'].replace('0 Ft', '999 Ft')
    graph = build_graph(retrieve_fn=lambda q, **kw: [hit, conflict],
                        generate_fn=lambda q, **kw: (_ for _ in ()).throw(AssertionError('Should not generate')),
                        rewrite_fn=lambda q: q)
    state = graph.invoke({'question': Q_PRECEDENCE, 'search_query': Q_PRECEDENCE,
                          'retrieval_round': 0, 'rewrite_count': 0, 'steps': []})
    assert state['result']['status'] == 'insufficient'
    assert state['result']['mode'] == 'duplicate_source_conflict'
    assert state['retrieval_round'] == 1
    assert 'grade:conflict' in state['steps']


def test_graph_no_followup_if_all_roles_present_when_langgraph_is_present():
    pytest.importorskip('langgraph')
    from src.graph import build_graph
    hits = needed_hits(Q_EXPIRY)
    from src.generation import answer_question
    graph = build_graph(retrieve_fn=lambda q, **kw: hits,
                        generate_fn=lambda q, hits_override: answer_question(q, hits_override=hits_override),
                        targeted_fn=lambda plan, h: (_ for _ in ()).throw(AssertionError('no followup')))
    state = graph.invoke({'question': Q_EXPIRY, 'search_query': Q_EXPIRY,
                          'retrieval_round': 0, 'rewrite_count': 0, 'steps': []})
    assert state['result']['status'] == 'partial'
    assert state['retrieval_round'] == 1
    assert 'grade:sufficient' in state['steps']

@pytest.mark.parametrize('question', [Q_EXPIRY, Q_DEBIT, Q_PRECEDENCE])
def test_shared_assembler_is_used_by_basic_and_agentic(question):
    from src.generation import answer_question
    hits = needed_hits(question)
    basic = answer_question(question, hits_override=hits)
    planned = answer_cross_document(plan_cross_document(question), hits)
    assert basic['mode'] == planned['mode']
    assert basic['status'] == planned['status']
    assert basic['answer'] == planned['answer']
    assert basic['sources'] == planned['sources']
