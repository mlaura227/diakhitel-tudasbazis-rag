from src.ui_helpers import diagnostic_json, history_as_csv, result_summary, source_line, status_label


def test_source_line_uses_supplied_metadata():
    assert source_line({"citation_id": "S1", "document_title": "Glosszárium", "page": "1", "section_id": "section-009"}) == (
        "[S1] Glosszárium | 1. oldal | section-009"
    )


def test_status_labels():
    assert status_label("answered") == "Megválaszolva"
    assert status_label("insufficient") == "Nem állapítható meg"


def test_result_summary():
    assert result_summary({"status": "answered", "mode": "extractive", "sources": [{"citation_id": "S1"}]}) == (
        "Megválaszolva · extractive · 1 forrás"
    )


def test_history_csv_has_two_comparable_rows_and_handles_commas():
    turn = {"question": "1,000 Ft?", "times": {"basic": 1.2, "agentic": 2.0}, "results": {
        "basic": {"status": "answered", "answer": "0 Ft", "mode": "extractive", "sources": [{"doc_id": "notice", "section_id": "abc"}]},
        "agentic": {"status": "insufficient", "answer": "Nem állapítható meg", "mode": "document_limit_guard", "sources": [],
                    "graph_trace": ["retrieve:1", "grade:sufficient", "answer:insufficient"], "retrieval_rounds": 1, "rewrite_count": 0},
    }}
    csv_data = history_as_csv([turn])
    assert csv_data.count("\n") == 3
    assert '"1,000 Ft?"' in csv_data
    assert "notice:abc" in csv_data
    assert "retrieve:1 -> grade:sufficient -> answer:insufficient" in csv_data


def test_diagnostics_excludes_full_content():
    raw = diagnostic_json({"status": "answered", "retrieval_trace": [], "content": "private source passage"})
    assert "private source passage" not in raw
    assert '"status": "answered"' in raw
