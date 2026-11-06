"""Source-backed development regressions"""
import copy
import pytest

from src.source_sections import lookup_sections
from src.faceted_qa import plan_facets, missing_facets
from src.generation import answer_question
from src.applicability import check_applicability

Q = ('Mennyi az OTP Diákhitel Számla számlavezetési díja, '
     'és mennyi a bankkártya pótlásának díja?')
KEYS = [
    ('fee_schedule_20260901', 'SRC-P03-B01'),
    ('notice_20251118', 'source-p1-card-fees'),
]


def test_facet_parser_preserves_two_independent_questions():
    p = plan_facets(Q)
    assert p and len(p.facets) == 2
    assert 'számlavezetési' in p.facets[0]
    assert 'pótlásának' in p.facets[1]
    assert 'Diákhitel' in p.facets[1]


def test_facet_parser_does_not_split_anaphoric_definition():
    assert plan_facets('A SEPA forintátutalás neve, ugye? Hogyan definiálja az OTP?') is None


def test_fully_evidenced_facet_assembly():
    hits = lookup_sections(KEYS)
    r = answer_question(Q, hits_override=hits)
    assert r['status'] == 'answered'
    assert r['mode'] == 'facet_coverage_assembly'
    assert len(r['sources']) == 2
    assert '0 Ft' in r['answer'] and 'díjmentes' in r['answer']
    by_id = {s['citation_id']: s for s in r['sources']}
    for item in r['evidence']:
        cited = by_id[item['source_id']]
        hit = next(h for h in hits if h['metadata']['section_id'] == cited['section_id'])
        assert item['quote'] in hit['content']


def test_missing_facet_downgrades_to_partial_not_answered():
    hits = lookup_sections(KEYS[:1])
    p = plan_facets(Q)
    assert len(missing_facets(p, hits)) == 1
    r = answer_question(Q, hits_override=hits)
    assert r['status'] == 'partial'
    assert len(r['missing_facets']) == 1
    assert 'nem áll rendelkezésre' in r['answer']
    assert len(r['sources']) == 1


def test_sepa_definition_plus_phone_fee():
    q = 'Az OTP Diákhitel Számla SEPA átutalása mit jelent, és mennyi a telefonos díja?'
    hits = lookup_sections([
        ('glossary_undated', 'section-009'),
        ('fee_schedule_20260901', 'SRC-P06-B01'),
    ])
    r = answer_question(q, hits_override=hits)
    assert r['status'] == 'answered'
    assert 'euróban' in r['answer']
    assert 'telefonon' in r['answer']
    assert len(r['sources']) == 2


def _fake_result(answer, title, quote, status='answered'):
    return dict(status=status, answer=answer, sources=[{
        'citation_id': 'S1', 'section_title': title,
        'doc_id':'fee_schedule_20260901', 'section_id':'TEST',
    }], evidence=[{'source_id':'S1', 'quote':quote}], mode='condition_checked')


def test_foreign_purchase_rejects_domestic_fee():
    r = _fake_result('A külföldi vásárlás díja 0 Ft.',
                     'Bankkártyás vásárlás belföldön',
                     '- Betéti kártyával történő belföldi vásárlás: 0 Ft.')
    checked = check_applicability('Mennyi a betéti kártyával külföldön végzett vásárlás díja?',r)
    assert checked['status'] == 'insufficient'
    assert checked['applicability_issue'] == 'domestic_tariff_for_foreign_purchase'
    assert '0 Ft' not in checked['answer']


def test_branch_withdrawal_rejects_atm_tariff():
    r = _fake_result('1 711 Ft + 0,9%', 'Készpénzfelvétel belföldön',
                     '| Betéti bankkártyával, saját ATM-ből | 150 000 Ft tranzakcióösszegig | 172 Ft + 1,79% |')
    checked = check_applicability('Mennyi a készpénzfelvétel díja saját OTP fiókban?',r)
    assert checked['status'] == 'insufficient'
    assert checked['applicability_issue'] == 'atm_tariff_for_branch_withdrawal'


def test_future_fee_not_derived_from_today_price():
    r = _fake_result('A 2028-as pótkártya díja 0 Ft.',
                     'Bankkártya éves díja',
                     '| Mastercard Online Diákhitel pótkártya díja | 0 Ft |')
    checked = check_applicability('Mennyi lesz a pótkártya díja 2028-ban?', r)
    assert checked['status'] == 'insufficient'
    assert checked['applicability_issue'] == 'future_fee_not_established'


def test_future_fee_preserves_explicit_uncertainty():
    r = _fake_result('A 2028-as pontos díj nem állapítható meg a jelenlegi díjsorból.',
                     'Bankkártya éves díja',
                     '| Mastercard Online Diákhitel pótkártya díja | 0 Ft |')
    checked = check_applicability('Mennyi lesz a pótkártya díja 2028-ban?', r)
    assert checked['status'] == 'answered'  # content validation is separate


def test_unrelated_existing_basic_answer_unchanged():
    q='Mennyibe kerül az OTP Diákhitel Számla számlavezetése?'
    r=answer_question(q, hits_override=lookup_sections(KEYS[:1]))
    assert r['status']=='answered' and r['mode'] != 'facet_coverage_assembly'


def test_graph_targets_only_missing_facet_when_langgraph_installed():
    pytest.importorskip('langgraph')
    from src.graph import build_graph
    all_hits = lookup_sections(KEYS)
    calls = []

    def retrieve(query, k=5, document_type=None, **kwargs):
        calls.append(query)
        if len(calls) == 1:
            return all_hits[:1]
        return all_hits[1:]

    graph = build_graph(retrieve_fn=retrieve, generate_fn=answer_question,
                        rewrite_fn=lambda q: q)
    state = graph.invoke({'question':Q, 'search_query':Q,
                          'retrieval_round':0, 'rewrite_count':0, 'steps':[],
                          'targeted_facets_done':False})
    assert state['result']['status'] == 'answered'
    assert state['retrieval_round'] == 2
    assert 'grade:weak' in state['steps']
    assert 'targeted_facets:1' in state['steps']
    assert 'grade:sufficient' in state['steps']
    assert len(calls)==2


def test_future_opening_deadline_plus_current_price_is_not_future_tariff():
    r = _fake_result('A 2027-es számlavezetési díj 0 Ft.',
                     'Számlavezetési díjak',
                     '- Elektronikus kivonattal: havi díj 0 Ft; teljes éves díj 0 Ft.')
    r['sources'].append({'citation_id':'S2', 'section_title':'Számlanyitási feltételek'})
    r['evidence'].append({'source_id':'S2', 'quote':'2027. június 30-áig nyitható a számla.'})
    assert check_applicability('2027-ben mennyi a havi számlavezetési díj?', r)['status'] == 'insufficient'


def test_legal_fallback_returns_quote_not_unsupported_complete_conclusion():
    from src.legal_fallback import answer_legal_fallback
    body = ('## OTP Diákhitel Számla – különös szabályok\n'
            'Az OTP Diákhitel Számla számlavezetése nem szüneteltethető.\n')
    hit = {'metadata':{'doc_id':'account_rules_20261015', 'section_id':'TEST-LEGAL',
                       'document_type':'account_regulation',
                       'section_title':'OTP Diákhitel Számla – különös szabályok',
                       'page':'33'},
           'content':body}
    r=answer_legal_fallback('Kérhetek Diákhitel Számla szüneteltetést?', [hit])
    assert r is not None
    assert r['mode']=='legal_quote_fallback'
    assert r['status']=='partial'
    assert 'nem szüneteltethető' in r['answer']
    assert r['evidence'][0]['quote'] in body


def test_legal_fallback_never_borrows_another_products_rule():
    from src.legal_fallback import answer_legal_fallback
    hit={'metadata':{'document_type':'account_regulation',
        'section_title':'Lakossági OKÉ Számla – különös rendelkezések',
        'doc_id':'account_rules_20261015', 'section_id':'XX'},
        'content':'## Lakossági OKÉ Számla\nAz OKÉ számla nem szüneteltethető.'}
    assert answer_legal_fallback('Szüneteltethetem az OTP Diákhitel Számlát?', [hit]) is None
