"""Offline (mocked) tests for the evidence-first baseline"""
import json
import sys
import types
from unittest.mock import patch

from src.extractive import extract_answer
from src.generation import answer_question, prepare_context, _money_tokens


def hit(doc_id, section_id, page, content, title, doc_type="fee_schedule", distance=0.2):
    return {
        "content": content,
        "metadata": {
            "doc_id": doc_id, "section_id": section_id,
            "document_title": doc_id, "document_type": doc_type,
            "section_title": title, "page": page,
        },
        "distance": distance,
    }


ACCOUNT = hit(
    "fee_schedule_20260901", "SRC-P03-B01", "3",
    "### [SRC-P03-B01] Számlavezetési díjak\n"
    "Az OTP Diákhitel Számla kedvezmény nélküli számlavezetési és csomagdíja:\n"
    "- Elektronikus kivonattal: havi díj 0 Ft; teljes éves díj 0 Ft.\n"
    "- Papíralapú kivonattal: havi díj 0 Ft; teljes éves díj 0 Ft.",
    "[SRC-P03-B01] Számlavezetési díjak",
)
COPY = hit(
    "fee_schedule_20260901", "SRC-P09-B03", "9",
    "### [SRC-P09-B03] Igazolások kiállítása és rendelkezésre bocsátása\n"
    "- Postai úton kiküldött rendszeres havi bankszámlakivonat másolata papíralapon: 1 000 Ft.",
    "[SRC-P09-B03] Igazolások kiállítása és rendelkezésre bocsátása",
)
CARD = hit(
    "notice_20251118", "source-p1-card-fees", "1",
    "## 2. A számlához kötelezően kapcsolódó bankkártya díjai és költségei\n"
    "| Mastercard Online Diákhitel pótkártya díja | díjmentes |",
    "2. A számlához kötelezően kapcsolódó bankkártya díjai és költségei",
    "product_notice",
)
SEPA = hit(
    "glossary_undated", "section-009", "1",
    "### 8. Átutalás euróban (SEPA)\n"
    "A számlavezető az ügyfél utasítására pénzt juttat el az ügyfél számlájáról "
    "egy másik számlára, belföldön és külföldön, euróban. [Forrás: 1. oldal]",
    "8. Átutalás euróban (SEPA)", "glossary",
)
REGULAR = hit(
    "glossary_undated", "section-007", "1",
    "### 6. Rendszeres átutalás\n"
    "A számlavezető az ügyfél utasítására rendszeresen, azonos összegben "
    "pénzt juttat el az ügyfél számlájáról egy másik számlára. [Forrás: 1. oldal]",
    "6. Rendszeres átutalás", "glossary",
)
ATM = hit(
    "fee_schedule_20260901", "SRC-P07-B04", "7",
    "### [SRC-P07-B04] Készpénzfelvétel belföldön\n"
    "| Eszköz és helyszín | Feltétel / megjegyzés | Díj |\n"
    "| Betéti bankkártyával, saját ATM-ből | 150 000 Ft tranzakcióösszegig | 172 Ft + 1,79% |",
    "[SRC-P07-B04] Készpénzfelvétel belföldön",
)
DEVIZA = hit(
    "fee_schedule_20260901", "SRC-P06-B03", "6",
    "### [SRC-P06-B03] Devizajóváírás\n"
    "| Külső számláról, eltérő devizában | 0,05%, minimum 8,59 EUR, maximum 41,23 EUR |",
    "[SRC-P06-B03] Devizajóváírás",
)


def llm_reply(status, answer, evidence, done_reason="stop"):
    return {
        "message": {"content": json.dumps({
            "status": status, "answer": answer, "evidence": evidence
        }, ensure_ascii=False)},
        "done_reason": done_reason,
    }


class StubClient:
    def __init__(self, replies):
        self.replies = list(replies)
        self.calls = []

    def chat(self, **kwargs):
        self.calls.append(kwargs)
        return self.replies.pop(0)


def run(question, hits, replies=()):
    stub = types.ModuleType("src.retrieval")
    stub.retrieve_documents = lambda question, k=5, document_type=None: [
        h for h in hits if document_type is None or h["metadata"]["document_type"] == document_type
    ]
    client = StubClient(replies)
    with patch.dict(sys.modules, {"src.retrieval": stub}):
        result = answer_question(question, client=client)
    return result, client


def test_context_metadata_is_preserved():
    context, refs = prepare_context([ACCOUNT, CARD])
    assert "[S1]" in context and "[S2]" in context
    assert refs["S1"]["metadata"]["section_id"] == "SRC-P03-B01"


def test_account_answer_is_extracted_without_ollama():
    result, client = run("Mennyibe kerül az OTP Diákhitel Számla számlavezetése?", [ACCOUNT])
    assert result["status"] == "answered"
    assert result["mode"] == "extractive"
    assert "havonta 0 Ft" in result["answer"]
    assert "évente 0 Ft" in result["answer"]
    assert len(result["evidence"]) == 2
    assert not client.calls


def test_paper_statement_and_copy_are_distinguished():
    result, client = run(
        "Ingyenes a papíralapú rendszeres számlakivonat vagy az utólag kért másolat?",
        [COPY, ACCOUNT],
    )
    assert result["status"] == "answered"
    assert "havonta 0 Ft" in result["answer"]
    assert "1 000 Ft" in result["answer"]
    assert {s["section_id"] for s in result["sources"]} == {"SRC-P09-B03", "SRC-P03-B01"}
    assert not client.calls


def test_paper_copy_alone_uses_its_own_fee_row():
    result, client = run("Mennyibe kerül a bankszámlakivonat másolata?", [COPY])
    assert result["status"] == "answered"
    assert "1 000 Ft" in result["answer"]
    assert "havonta 0 Ft" not in result["answer"]
    assert not client.calls


def test_paper_statement_comparison_needs_both_sources():
    assert extract_answer(
        "Ingyenes a papíralapú rendszeres számlakivonat vagy az utólag kért másolat?",
        [COPY],
    ) is None


def test_card_replacement_uses_notice_not_issuance_fee():
    result, client = run("Mennyibe kerül a Mastercard Online Diákhitel kártya pótlása?", [ACCOUNT, CARD])
    assert result["status"] == "answered"
    assert "díjmentes" in result["answer"]
    assert "0 Ft" not in result["answer"]
    assert result["sources"][0]["section_id"] == "source-p1-card-fees"
    assert not client.calls


def test_sepa_definition_uses_correct_glossary_section():
    result, client = run("Mit jelent a SEPA átutalás?", [REGULAR, SEPA])
    assert result["status"] == "answered"
    assert result["sources"][0]["section_id"] == "section-009"
    assert "euróban" in result["answer"]
    assert "belföldön és külföldön" in result["answer"]
    assert "rendszeresen" not in result["answer"]
    assert not client.calls


def test_general_glossary_definition_is_source_extractive():
    result, _ = run("Mit jelent a rendszeres átutalás?", [SEPA, REGULAR])
    assert result["status"] == "answered"
    assert result["sources"][0]["section_id"] == "section-007"


def test_sepa_not_answered_using_regular_transfer_quote_even_if_quote_is_real():
    q = "Mit jelent a SEPA átutalás?"
    wrong_quote = REGULAR["content"].splitlines()[1].split(" [Forrás:")[0]
    bad = llm_reply("answered", "A SEPA rendszeres, azonos összegű átutalás.", [
        {"source_id": "S1", "quote": wrong_quote}
    ])
    result, client = run(q, [REGULAR], [bad, bad])
    assert result["status"] == "insufficient"
    assert len(client.calls) == 2


def test_atm_above_documented_limit_abstains_without_model():
    result, client = run("Mennyi a díja egy 300 000 Ft-os saját ATM-es készpénzfelvételnek?", [ATM])
    assert result["status"] == "insufficient"
    assert "150 000 Ft" in result["answer"]
    assert "300 000 Ft" in result["answer"]
    assert result["sources"][0]["section_id"] == "SRC-P07-B04"
    assert result["mode"] == "document_limit_guard"
    assert not client.calls


def test_no_fees_are_hardcoded_in_extractive_engine():
    altered = {**ACCOUNT, "content": ACCOUNT["content"].replace("0 Ft", "25 Ft")}
    result, _ = run("Mennyibe kerül a számlavezetés?", [altered])
    assert result["status"] == "answered"
    assert "havonta 25 Ft" in result["answer"]
    assert "évente 25 Ft" in result["answer"]


def test_fallback_generated_answer_with_matching_exact_quote():
    quote = DEVIZA["content"].splitlines()[1]
    result, client = run("Milyen díja van a devizajóváírásnak?", [DEVIZA], [
        llm_reply("answered", "Eltérő devizában a díj 0,05%, minimum 8,59 EUR, maximum 41,23 EUR.", [
            {"source_id": "S1", "quote": quote}
        ])
    ])
    assert result["status"] == "answered"
    assert result["mode"] == "generative_with_literal_quote_checks"
    assert len(client.calls) == 1


def test_fallback_refuses_invented_numeric_fee():
    quote = DEVIZA["content"].splitlines()[1]
    wrong = llm_reply("answered", "A devizajóváírás 9 000 Ft.", [
        {"source_id": "S1", "quote": quote}
    ])
    result, client = run("Mennyi a devizajóváírás díja?", [DEVIZA], [wrong, wrong])
    assert result["status"] == "insufficient"
    assert len(client.calls) == 2


def test_no_hits_abstains():
    result, client = run("Mennyi a távoli költség?", [], [])
    assert result["status"] == "insufficient"
    assert not client.calls


def test_money_token_normalization():
    assert _money_tokens("1 000 Ft, 0,65%, 5 131 Ft") == {"1000ft", "0.65%", "5131ft"}
