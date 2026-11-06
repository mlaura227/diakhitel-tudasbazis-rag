"""
Conservative basic RAG: cite literal source quotes or abstain.
Drop-in replacement for src/generation.py. Does not modify ChromaDB.
"""
from __future__ import annotations

import copy
import json
import re
import unicodedata
from typing import Any

from src.config import OLLAMA_BASE_URL
from src.extractive import extract_answer, norm
from src.query_policy import plan_query, scope_result
from src.routed_retrieval import retrieve_routed
from src.answer_router import answer_from_sources
from src.guards import check_documented_atm_ceiling, check_paper_copy_fee_claim
from src.prompts import ANSWER_SCHEMA, SYSTEM_PROMPT

LLM_MODEL = "qwen2.5:3b"
TOP_K = 5
MAX_CONTEXT_CHARS = 5500
RETRY_CONTEXT_CHARS = 4500


def prepare_context(
    hits: list[dict[str, Any]], max_chars: int = MAX_CONTEXT_CHARS
) -> tuple[str, dict[str, dict[str, Any]]]:
    """Add stable local IDs, but never truncate the source tables."""
    parts: list[str] = []
    references: dict[str, dict[str, Any]] = {}
    used = 0
    seen: set[tuple[str, str]] = set()
    for hit in hits:
        meta = hit["metadata"]
        key = (str(meta.get("doc_id", "")), str(meta.get("section_id", "")))
        if key in seen:
            continue
        seen.add(key)
        sid = f"S{len(references) + 1}"
        raw = str(hit["content"]).strip()
        passage = (
            f"[{sid}] {meta.get('document_title', 'Ismeretlen forrás')}\n"
            f"Oldal: {meta.get('page', 'ismeretlen')}\n"
            f"Szakasz: {meta.get('section_id', 'ismeretlen')}\n"
            f"Típus: {meta.get('document_type', 'ismeretlen')}\n"
            f"Szöveg:\n{raw}"
        )
        if used + len(passage) + 2 > max_chars:
            continue
        references[sid] = {"metadata": dict(meta), "content": raw}
        parts.append(passage)
        used += len(passage) + 2
    return "\n\n".join(parts), references


def _field(value: Any, name: str, default: Any = None) -> Any:
    return value.get(name, default) if isinstance(value, dict) else getattr(value, name, default)


def _parse_response(response: Any) -> dict[str, Any]:
    if _field(response, "done_reason", "stop") == "length":
        raise ValueError("Model output was cut off (done_reason=length)")
    raw = _field(_field(response, "message"), "content", "")
    if not isinstance(raw, str) or not raw.strip():
        raise ValueError("Empty output")
    raw = raw.strip()
    if raw.startswith("```json") and raw.endswith("```"):
        raw = raw[7:-3].strip()
    value = json.loads(raw)
    if not isinstance(value, dict):
        raise ValueError("JSON root is not an object")
    if value.get("status") not in {"answered", "partial", "insufficient"}:
        raise ValueError("Missing/invalid status")
    if not isinstance(value.get("answer"), str) or not isinstance(value.get("evidence"), list):
        raise ValueError("Missing/invalid answer or evidence")
    if any(
        not isinstance(item, dict)
        or not isinstance(item.get("source_id"), str)
        or not isinstance(item.get("quote"), str)
        for item in value["evidence"]
    ):
        raise ValueError("Invalid evidence array")
    return value


def _norm(text: str) -> str:
    return " ".join(unicodedata.normalize("NFC", text).casefold().split())


def _money_tokens(text: str) -> set[str]:
    """Only a basic numeric cross-check; not a semantic entailment verifier."""
    regex = re.compile(
        r"(?<!\w)(\d{1,3}(?:[\s\u00a0]\d{3})+|\d+)([,.]\d+)?\s*(Ft|EUR|%)",
        re.IGNORECASE,
    )
    values: set[str] = set()
    for m in regex.finditer(text):
        number = re.sub(r"[\s\u00a0]", "", m.group(1))
        fractional = (m.group(2) or "").replace(",", ".")
        values.add(f"{number}{fractional}{m.group(3).casefold()}")
    return values


def _validate_evidence(
    parsed: dict[str, Any], references: dict[str, dict[str, Any]]
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    evidence = parsed["evidence"]
    if parsed["status"] in {"answered", "partial"} and not evidence:
        raise ValueError("Answer has no citations/evidence")
    if parsed["status"] == "insufficient" and evidence:
        raise ValueError("Insufficient answer unexpectedly claims supporting evidence")

    seen = set()
    validated: list[dict[str, Any]] = []
    sources: list[dict[str, Any]] = []
    for item in evidence:
        sid, quote = item["source_id"], item["quote"].strip()
        if sid not in references:
            raise ValueError(f"Invalid source ID: {sid}")
        if not (12 <= len(quote) <= 900):
            raise ValueError("Evidence quote is too short or too long")
        if _norm(quote) not in _norm(references[sid]["content"]):
            raise ValueError(f"Quote does not occur verbatim in source {sid}")
        validated.append({"source_id": sid, "quote": quote})
        if sid not in seen:
            sources.append({"citation_id": sid, **references[sid]["metadata"]})
            seen.add(sid)

    if parsed["status"] in {"answered", "partial"}:
        answer_tokens = _money_tokens(parsed["answer"])
        quote_tokens = _money_tokens("\n".join(e["quote"] for e in validated))
        extra = answer_tokens - quote_tokens
        if extra:
            raise ValueError(f"Numerical claims not found in quoted evidence: {sorted(extra)}")
    return validated, sources



def _require_relevant_source(
    question: str, parsed: dict[str, Any], references: dict[str, dict[str, Any]]
) -> None:
    """Reject obvious topic/citation mismatch (not a full entailment proof)."""
    if parsed["status"] == "insufficient":
        return
    cited = [references[item["source_id"]] for item in parsed["evidence"]]
    # Explicit product/technical acronyms must occur in a cited source section, not merely elsewhere among all retrieved passages. Guards against, e.g., citing the generic "rendszeres átutalás" definition for SEPA.
    acronyms = re.findall(r"\b[A-ZÁÉÍÓÖŐÚÜŰ]{3,}\b", question)
    for acronym in acronyms:
        wanted = norm(acronym)
        if not any(
            wanted in norm(str(ref["metadata"].get("section_title", "")))
            or wanted in norm(ref["content"])
            for ref in cited
        ):
            raise ValueError(f"Cited evidence does not concern requested topic: {acronym}")
    # Prevent using a glossary section about the wrong definition when the question explicitly names the target and a better glossary match exists.
    if _definition_question(question):
        glossary = [ref for ref in cited if ref["metadata"].get("document_type") == "glossary"]
        if not glossary:
            raise ValueError("Definition question cites no glossary definition")


def _trace(hits: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        {
            "doc_id": h["metadata"].get("doc_id"),
            "section_id": h["metadata"].get("section_id"),
            "page": h["metadata"].get("page"),
            "distance": h.get("distance"),
            "method": h.get("retrieval_method", "vector"),
        }
        for h in hits
    ]


def _definition_question(question: str) -> bool:
    q = question.casefold()
    return (
        "mit jelent" in q
        or "mi az a " in q
        or "fogalma" in q
        or "definíciója" in q
        or "definiáld" in q
    )


def _answer_question_core(
    question: str,
    k: int = TOP_K,
    client: Any = None,
    *,
    hits_override: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Retrieve, answer and mechanically verify evidence references.

    Does not guarantee semantic correctness: inspect outputs and use gold tests.
    """
    question = question.strip()
    if not question:
        raise ValueError("Question must not be empty")

    plan = plan_query(question)
    if plan.scope != "in_scope":
        return scope_result(plan)

    # Injected model clients exercise the original JSON validation path; the production pipeline uses condition-aware extraction first.
    if hits_override is not None:
        hits = hits_override
    elif client is not None:
        from src.retrieval import retrieve_documents
        hits = (retrieve_documents(question, k=k, document_type="glossary")
                if _definition_question(question) else retrieve_documents(question, k=k))
    else:
        hits = retrieve_routed(question, plan, k=k)
    trace = _trace(hits)
    if not hits:
        return {"status": "insufficient",
                "answer": "Nem találtam a kérdéshez felhasználható forrásrészletet.",
                "sources": [], "evidence": [], "retrieval_trace": trace,
                "mode": "no_evidence"}

    # Multi-document evidence assembly is SHARED by Basic and Agentic RAG.
    # Basic can only use sections already retrieved; LangGraph can fetch missing roles in a bounded targeted second lookup. This keeps answer construction identical in both engines for a defensible comparison.
    if client is None:
        from src.cross_document import plan_cross_document, answer_cross_document
        cross = plan_cross_document(question)
        if cross:
            result = answer_cross_document(cross, hits)
            result["retrieval_trace"] = trace
            return result

    guarded = check_documented_atm_ceiling(question, hits)
    if guarded:
        guarded["retrieval_trace"] = trace
        guarded["mode"] = "document_limit_guard"
        return guarded

    if client is None:
        extracted = answer_from_sources(question, plan, hits)
    else:
        extracted = extract_answer(question, hits)
    if extracted:
        extracted["retrieval_trace"] = trace
        return extracted

    # Contractual questions must not fall through to a weak quote-matching LLM.
    # Their clause-completeness validator returns a justified refusal when
    # required sections are missing. A separate agentic step can fetch them.
    if client is None:
        from src.legal_reasoning import plan_legal, answer_legal
        legal_plan = plan_legal(question)
        if legal_plan:
            legal = answer_legal(question, hits, legal_plan)
            if legal is not None:
                legal["retrieval_trace"] = trace
                return legal

    # For contractual paraphrases outside the finite legal plans, return a
    # *partial, verbatim* regulation excerpt rather than either hallucinating
    # a complete legal conclusion or hiding relevant evidence entirely.
    if client is None:
        from src.legal_fallback import answer_legal_fallback
        quote_based = answer_legal_fallback(question, hits)
        if quote_based:
            quote_based["retrieval_trace"] = trace
            return quote_based

    # Unknown monetary conditions must not fall through to LLM tariff guesses.
    if client is None and ("fee" in plan.intents or "definition" in plan.intents):
        return {
            "status": "insufficient", "sources": [], "evidence": [],
            "answer": "Nem található minden megadott feltételhez egyértelműen alkalmazható díjsor vagy fogalommeghatározás a betöltött dokumentumokban.",
            "mode": "condition_not_resolved", "retrieval_trace": trace, "attempts": 0,
        }

    if client is None:
        from ollama import Client
        client = Client(host=OLLAMA_BASE_URL)

    diagnostics: list[str] = []
    for attempt, (context_budget, output_tokens) in enumerate(
        [(MAX_CONTEXT_CHARS, 650), (RETRY_CONTEXT_CHARS, 850)], start=1
    ):
        context, references = prepare_context(hits, max_chars=context_budget)
        if not references:
            diagnostics.append(f"Attempt {attempt}: no source fits context")
            continue
        # Constrain model-generated IDs to the references actually provided.
        schema = copy.deepcopy(ANSWER_SCHEMA)
        schema["properties"]["evidence"]["items"]["properties"]["source_id"]["enum"] = list(references)
        response = client.chat(
            model=LLM_MODEL,
            stream=False,
            format=schema,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": (
                        f"KÉRDÉS:\n{question}\n\nFORRÁSRÉSZLETEK:\n{context}\n\n"
                        "Válaszolj kizárólag JSON-nal. Ha answer/partial, adj legalább "
                        "egy tényleges szó szerinti idézetet az evidence mezőben. "
                        "Kérlek, ellenőrizd a szolgáltatás és a díj feltételeit."
                    ),
                },
            ],
            options={"temperature": 0, "seed": 42, "num_ctx": 4096, "num_predict": output_tokens},
        )
        try:
            parsed = _parse_response(response)
            evidence, sources = _validate_evidence(parsed, references)
            _require_relevant_source(question, parsed, references)
            check_paper_copy_fee_claim(question, parsed["answer"], hits)
            answer = parsed["answer"].strip()
            if parsed["status"] == "insufficient":
                # Don't display unverified claims even when the model abstains.
                answer = "A rendelkezésre álló forrásokból nem állapítható meg megbízható válasz."
            if not answer:
                raise ValueError("Empty answer")
            return {
                "status": parsed["status"],
                "answer": answer,
                "sources": sources,
                "evidence": evidence,
                "retrieval_trace": trace,
                "attempts": attempt,
                "mode": "generative_with_literal_quote_checks",
            }
        except (ValueError, TypeError) as exc:
            diagnostics.append(
                f"Attempt {attempt}: {type(exc).__name__}: {exc}; "
                f"done_reason={_field(response, 'done_reason', '?')}"
            )

    return {
        "status": "insufficient",
        "answer": (
            "Nem sikerült a választ megbízható, szó szerinti forrásidézettel "
            "alátámasztani. A rendszer ezért nem ad meg ellenőrizetlen díjat vagy feltételt."
        ),
        "sources": [],
        "evidence": [],
        "retrieval_trace": trace,
        "diagnostics": diagnostics,
        "mode": "generative_validation_failed",
    }


def answer_question(
    question: str,
    k: int = TOP_K,
    client: Any = None,
    *,
    hits_override: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Shared facet-coverage + applicability validation for both engines.

    It preserves the existing core path for single-part questions and for
    injected/mock model clients. A separate graph may acquire missing facets.
    """
    from src.applicability import check_applicability
    from src.faceted_qa import plan_facets, answer_facets
    # Don't duplicate retrieval: use the same source collection as the core.
    facets = plan_facets(question) if client is None else None
    if facets:
        from src.cross_document import plan_cross_document
        from src.legal_reasoning import plan_legal
        if plan_cross_document(question) or plan_legal(question):
            facets = None  # Preserve the established source-role planner.
    if facets:
        if hits_override is None:
            from src.routed_retrieval import retrieve_routed
            hits = retrieve_routed(question, plan_query(question), k=k)
        else:
            hits = hits_override
        result = answer_facets(facets, hits)
        result['retrieval_trace'] = _trace(hits)
        return result
    result = _answer_question_core(question, k, client, hits_override=hits_override)
    return check_applicability(question, result)
