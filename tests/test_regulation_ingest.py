from collections import Counter
from src.config import COLLECTION_NAME, resolve_source_path
from src.document_loader import load_documents, split_documents
from src.query_policy import plan_query
from src.routed_retrieval import retrieve_routed
from src.source_sections import lookup_sections


def test_regulation_sources_present():
    assert resolve_source_path('USZ_LAK_Bankszamla_20261015-knowledge.md').is_file()
    assert resolve_source_path('USZ_LAK_Bankszamla_20261015-rag.md').is_file()
    assert COLLECTION_NAME == 'otp_diakhitel_regulation_v2'


def test_all_133_sections_loaded():
    docs = split_documents(load_documents())
    counts = Counter(d.metadata['doc_id'] for d in docs)
    assert counts == {
        'fee_schedule_20260901': 29,
        'notice_20251118': 12,
        'glossary_undated': 21,
        'account_rules_20261015': 71,
    }
    assert len(docs) == 133
    ids = {(d.metadata['doc_id'], d.metadata['section_id']) for d in docs}
    assert len(ids) == len(docs)


def test_diakhitel_rules_retains_scope_and_page():
    docs = load_documents()
    matches = [d for d in docs if d.metadata['doc_id'] == 'account_rules_20261015'
               and 'termékcél, megnyitási határidő' in d.metadata['section_title']]
    assert len(matches) == 1
    d = matches[0]
    assert d.metadata['page'] == '33'
    assert d.metadata['effective_from'] == '2026-10-15'
    assert d.metadata['source_representation'] == 'interpreted_knowledge'
    assert d.metadata['source_transcription_file'].endswith('-rag.md')
    assert '*Forrás:' in d.page_content


def test_old_structured_tariff_lookup_unchanged():
    matches = lookup_sections([('fee_schedule_20260901', 'SRC-P03-B01')])
    assert len(matches) == 1
    assert '0 Ft' in matches[0]['content']


def test_routing_regulation_only_when_relevant():
    calls = []
    def fake_vector(question, k=5, document_type=None, document_types=None):
        calls.append((document_type, document_types))
        return []
    q = 'Milyen feltételekkel nyitható az OTP Diákhitel Számla?'
    retrieve_routed(q, plan_query(q), vector_fn=fake_vector)
    assert ('account_regulation', None) in calls
    calls.clear()
    q = 'Mennyibe kerül az OTP Diákhitel Számla számlavezetése?'
    retrieve_routed(q, plan_query(q), vector_fn=fake_vector)
    assert ('account_regulation', None) not in calls
    assert (None, ('fee_schedule','product_notice')) in calls
