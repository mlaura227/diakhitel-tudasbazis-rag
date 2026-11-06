import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.graph import grade_retrieval, safe_search_rewrite, expected_sections, merge_hits


def hit(doc, sec, text="Test passage"):
    return {
        "metadata": {"doc_id": doc, "section_id": sec, "document_type": "glossary" if doc == "glossary_undated" else "fee_schedule"},
        "content": text,
        "distance": 0.5,
    }


def test_grade_account_maintenance():
    good, _ = grade_retrieval("Mennyibe kerül a számlavezetés?", [hit("fee_schedule_20260901", "SRC-P03-B01")])
    assert good


def test_paper_copy_requires_both_sections():
    q = "Ingyenes a papíralapú rendszeres számlakivonat vagy az utólag kért másolat?"
    acc = hit("fee_schedule_20260901", "SRC-P03-B01")
    copy = hit("fee_schedule_20260901", "SRC-P09-B03")
    assert not grade_retrieval(q, [acc])[0]
    assert grade_retrieval(q, [acc, copy])[0]


def test_sepa_requires_correct_section():
    q = "Mit jelent a SEPA átutalás?"
    wrong = hit("glossary_undated", "section-007")
    right = hit("glossary_undated", "section-009")
    assert not grade_retrieval(q, [wrong])[0]
    assert grade_retrieval(q, [right])[0]


def test_rewrite_does_not_lose_amount():
    q = "Mennyi egy 300 000 Ft-os saját ATM készpénzfelvétel?"
    assert safe_search_rewrite(q, "saját ATM készpénzfelvétel") == q
    assert "300 000" in safe_search_rewrite(q, "300 000 Ft saját ATM díj")


def test_merge_deduplicates_and_keeps_order():
    a, b = hit("a", "x"), hit("b", "y")
    assert merge_hits([a], [a, b]) == [a, b]


def test_card_replacement_required_section():
    assert expected_sections("Mennyibe kerül a bankkártya pótlása?") == {
        ("notice_20251118", "source-p1-card-fees")
    }


def test_amount_guard_with_decimal_fees():
    q = "100,5 EUR a SEPA utalás?"
    assert safe_search_rewrite(q, "SEPA 100 EUR") == q


def test_graph_conditional_routing_if_langgraph_installed():
    try:
        import langgraph  # noqa: F401
    except ImportError:
        import pytest
        pytest.skip("LangGraph is not installed in this offline test runtime")

    from src.graph import build_graph
    calls = []
    needed = hit("fee_schedule_20260901", "SRC-P03-B01")

    def retrieve(q, **kwargs):
        calls.append(q)
        return [needed] if "számlavezet" in q else []

    def generate(q, **kwargs):
        return {"status": "answered", "answer": "0 Ft", "mode": "extractive", "sources": [], "evidence": []}

    g = build_graph(retrieve_fn=retrieve, generate_fn=generate, rewrite_fn=lambda q: q)
    result = g.invoke({
        "question": "Mennyibe kerül a számlavezetés?",
        "search_query": "Mennyibe kerül a számlavezetés?",
        "rewrite_count": 0, "retrieval_round": 0, "steps": []
    })
    assert result["result"]["status"] == "answered"
    assert result["retrieval_round"] == 1
    assert result["rewrite_count"] == 0
