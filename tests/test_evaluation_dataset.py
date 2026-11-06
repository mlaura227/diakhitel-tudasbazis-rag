from evaluation.evaluate import load_cases, make_row, summary, sections_from_sources
from evaluation.retrieval_eval import score


def test_dataset_size_and_splits():
    cases = load_cases()
    assert len(cases) == 44
    assert len({r['id'] for r in cases}) == 44
    assert {r['split'] for r in cases} == {'test'}
    assert len([r for r in cases if r['category'] == 'false_premise']) >= 5
    assert len([r for r in cases if r['category'] == 'other_bank']) >= 3
    assert len([r for r in cases if r['category'] == 'out_of_scope']) >= 5


def test_scope_cases_have_no_gold_sections():
    for case in load_cases():
        if case['category'] in {'other_bank', 'out_of_scope'}:
            assert case['expected_status'] == 'insufficient'
            assert case['gold_relevant_sections'] == []


def test_false_premise_requires_correction():
    for case in load_cases():
        if case['category'] == 'false_premise':
            assert case['expected_outcome'] == 'correct_false_premise'
            assert case['expected_status'] == 'answered'
            assert case['gold_relevant_sections']


def test_retrieval_score():
    score_ = score(['doc:a', 'doc:b'], ['doc:a','other:q', 'doc:b'], 3)
    assert score_['precision_at_k'] == 2/3
    assert score_['recall_at_k'] == 1.0
    assert score_['hit_at_1']
    assert score_['mrr_at_k'] == 1.0
    assert score([], ['doc:a'], 1) is None


def test_subprocess_timeout_row_is_not_success():
    case = load_cases()[0]
    row = make_row(case, 'basic', {'kind':'timeout', 'elapsed_seconds': 75.1})
    assert row['observed_status'] == 'timeout'
    assert row['timed_out']
    assert not row['status_match']
    assert row['human_answer_correct'] == ''


def test_result_citation_coverage():
    case = load_cases()[0]
    ref = case['gold_relevant_sections'][0]
    doc, section = ref.split(':', 1)
    row = make_row(case, 'basic', {'kind':'ok','elapsed_seconds': 2.0,'result': {
        'status':'answered', 'answer':'Sample',
        'sources':[{'doc_id':doc,'section_id':section}],
    }})
    assert row['status_match']
    assert row['gold_source_coverage'] == 1.0
    assert '2.0' in summary([row])
