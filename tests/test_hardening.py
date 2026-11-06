import copy
import pytest
from src.query_policy import plan_query
from src.source_sections import lookup_sections
from src.generation import answer_question
from src.tariff_rules import resolve_fee


def answer(q: str):
    plan=plan_query(q)
    return answer_question(q, hits_override=lookup_sections(plan.sections))


@pytest.mark.parametrize('q', [
    'A K&H ifjúsági számlán mennyi a havi számlavezetési díj?',
    'Mekkora az Erste diákszámla bankkártyájának éves díja?',
    'Hány euró a Revolutnál a SEPA átutalás díja?',
    'Az MBH Bank ATM-je mennyit számít fel MBH-s ügyfeleknek?',
    'OTP Diákhitel vagy Raiffeisen diákszámla: melyik olcsóbb?',
    'Szia mit jelent az otp?',
    'Mit főzzek holnap vacsorára?',
])
def test_scope_exit_without_retrieval(q):
    r=answer_question(q)
    assert r['status']=='insufficient'
    assert r['mode']=='scope_gate'
    assert not r['sources']


def test_in_scope_otp_card_at_external_atm_not_cross_bank_claim():
    q='Belföldön az MBH ATM-jéből OTP Diákhitel betéti kártyával mennyi a készpénzfelvétel díja?'
    assert plan_query(q).scope=='in_scope'


def test_domestic_foreign_atm_not_interchanged():
    r=answer('Belföldön idegen bank ATM-jéből mennyibe kerül betéti kártyával készpénzt felvenni?')
    assert r['status']=='answered'
    assert '1 711 Ft + 0,9%' in r['answer']
    assert '3 145' not in r['answer']
    assert r['sources'][0]['section_id']=='SRC-P07-B04'


def test_euro_eea_foreign_atm_refuses():
    r=answer('Mennyibe kerül eurót felvenni Németországban egy idegen ATM-ből?')
    assert r['status']=='insufficient'
    assert '3 145' not in r['answer']
    assert r['sources'][0]['section_id']=='SRC-P08-B01'


def test_own_atm_cap_cannot_be_extrapolated():
    r=answer('Mennyi a díja egy 275 000 Ft-os saját ATM-es készpénzfelvételnek?')
    assert r['status']=='insufficient'
    assert '150 000 Ft' in r['answer']


def test_full_composite_tariff_is_preserved():
    r=answer('A legfeljebb 150 000 Ft-os saját ATM-es betéti kártyás készpénzfelvétel díja mennyi?')
    assert r['status']=='answered'
    assert '172 Ft + 1,79%' in r['answer']


def test_fee_question_never_returns_glossary_definition():
    r=answer('Mi a fedezetigazolás kiállításának feltüntetett díja?')
    assert r['status']=='answered'
    assert '1 000 Ft' in r['answer']
    assert r['sources'][0]['section_id']=='SRC-P09-B03'


def test_compound_question_requires_definition_and_fee():
    r=answer('Mi az eurós SEPA utalás, és van-e költsége, ha személyesen a fiókban adom be?')
    assert r['status']=='answered'
    citations={s['doc_id'] for s in r['sources']}
    assert 'glossary_undated' in citations
    assert 'fee_schedule_20260901' in citations
    assert '0,65%' in r['answer']


def test_sms_monthly_vs_per_message_fees():
    r=answer('Ha a számlám és az SMS-csomag havi díja nulla, az értesítő SMS is ingyenes a Telekomnál?')
    assert r['status']=='answered'
    assert '0 Ft' in r['answer']
    assert '61 Ft' in r['answer']
    assert len(r['sources']) >= 2


def test_card_false_premise_not_recited():
    r=answer('Úgy tudom, az OTP Diákhitel pótkártya kiállítása 1 500 Ft. Jól tudom?')
    assert r['status']=='answered'
    assert 'díjmentes' in r['answer']
    assert '1 500 Ft' not in r['answer']


def test_source_derived_tariff_changes_with_source_row():
    q='Mennyibe kerül belföldön idegen ATM-ből betéti kártyával készpénzt felvenni?'
    h=copy.deepcopy(lookup_sections((('fee_schedule_20260901','SRC-P07-B04'),))[0])
    h['content']=h['content'].replace('1 711 Ft + 0,9%', '2 222 Ft + 1,2%')
    r=resolve_fee(q,[h])
    assert r['status']=='answered'
    assert '2 222 Ft + 1,2%' in r['answer']
    assert r['evidence'][0]['quote'] in h['content']


def test_no_zero_inference_from_not_applicable():
    r=answer('Az ATM-limitmódosításnál a díjjegyzék konkrétan 0 Ft-ot ír?')
    assert r['status']=='answered'
    assert 'Díjtípust nem alkalmazzuk' in r['answer']
    assert 'nem azonos' in r['answer']


def test_sepa_definition_corrects_huf_false_premise():
    r=answer('A SEPA forintátutalás neve, ugye? Hogyan definiálja az OTP?')
    assert r['status']=='answered'
    assert 'nem forintátutalás' in r['answer']
    assert 'euróban' in r['answer']


def test_document_date_not_conflated_with_effective_date():
    r=answer('Biztosan 2026. szeptember 1-jén lépett jogilag hatályba a díjjegyzék?')
    assert r['status']=='insufficient'
    assert r['sources'][0]['section_id']=='section-001'
