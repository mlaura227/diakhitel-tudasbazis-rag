"""
Bounded LangGraph retrieval workflow for the OTP Diákhitel RAG project.
This *does not* replace the Basic RAG baseline. It adds conditional retrieval, query rewriting (for search only) and bounded retries around the existing
source-grounded generation function. It makes no claim that keyword grading or literal quotations prove semantic entailment.
"""
from __future__ import annotations

import re
import unicodedata
from typing import Any, Callable, TypedDict
from src.query_policy import plan_query, scope_result
from src.routed_retrieval import retrieve_routed

Hit = dict[str, Any]
RetrieveFn = Callable[..., list[Hit]]
GenerateFn = Callable[..., dict[str, Any]]
RewriteFn = Callable[[str], str]

MAX_REWRITES = 1
TOP_K = 5


class RAGState(TypedDict, total=False):
    question: str
    search_query: str
    hits: list[Hit]
    previous_hits: list[Hit]
    retrieval_round: int
    rewrite_count: int
    relevance: str
    relevance_reason: str
    result: dict[str, Any]
    steps: list[str]
    scope: str
    targeted_legal_done: bool
    conflict_ids: list[str]
    missing_facets: list[str]
    targeted_facets_done: bool


def normalize(text: str) -> str:
    text = unicodedata.normalize("NFKD", text.casefold())
    return "".join(c for c in text if not unicodedata.combining(c))


def expected_sections(question: str) -> set[tuple[str, str]]:
    """Small, transparent domain checks for regression-critical topics."""
    q = normalize(question)
    if "kivonat" in q and ("masolat" in q or "utolag" in q):
        answer = {("fee_schedule_20260901", "SRC-P09-B03")}
        if "vagy" in q or "rendszeres" in q or "papir" in q:
            answer.add(("fee_schedule_20260901", "SRC-P03-B01"))
        return answer
    if ("szamlavezet" in q or "szamladij" in q) and not "kivonat masolat" in q:
        return {("fee_schedule_20260901", "SRC-P03-B01")}
    if "kartya" in q and any(word in q for word in ("potlas", "potkartya", "potol")):
        return {("notice_20251118", "source-p1-card-fees")}
    if "sepa" in q and any(word in q for word in ("mit jelent", "fogalma", "mi az", "definici")):
        return {("glossary_undated", "section-009")}
    return set()


def is_definition(question: str) -> bool:
    q = normalize(question)
    return any(word in q for word in ("mit jelent", "mi az a ", "fogalma", "definici", "definial"))


def grade_retrieval(question: str, hits: list[Hit]) -> tuple[bool, str]:
    """Evidence-role completeness gate; not a semantic entailment verifier."""
    from src.cross_document import plan_cross_document, missing_roles as missing_cross
    cross = plan_cross_document(question)
    if cross:
        missing = missing_cross(cross, hits)
        if missing:
            return False, "Hiányzó többdokumentumos bizonyítékok: " + ", ".join(missing)
        return True, "A többdokumentumos feladat összes forrásszerepe rendelkezésre áll."
    if not hits:
        return False, "Nem érkezett találat."
    actual = {(str(h.get("metadata", {}).get("doc_id", "")),
               str(h.get("metadata", {}).get("section_id", ""))) for h in hits}
    from src.legal_reasoning import plan_legal, missing_roles
    legal = plan_legal(question)
    if legal:
        missing = missing_roles(legal, hits)
        if missing:
            return False, "Hiányzó jogi szakaszok: " + ", ".join(missing)
        return True, "A jogi kérdés összes tervezett forrásszakasza rendelkezésre áll."
    required = expected_sections(question)
    if required:
        missing = sorted(required - actual)
        if missing:
            return False, f"Hiányzó szakasz(ok): {missing}"
        return True, "Megtaláltuk a felismert kérdéshez szükséges szakaszokat."
    if is_definition(question):
        if not any(h.get("metadata", {}).get("document_type") == "glossary" for h in hits):
            return False, "Definíciós kérdéshez nem találtunk glosszáriumrészt."
    return True, "Találat rendelkezésre áll; szemantikus megfelelés később ellenőrizendő."


def _question_numbers(text: str) -> set[str]:
    return set(re.findall(r"\d+(?:[\s\u00a0]\d{3})*(?:[.,]\d+)?", text))


def safe_search_rewrite(original: str, candidate: str) -> str:
    """Never allow a rewritten search query to drop explicit numeric conditions.

    The actual answer always receives *original*, never the rewritten string.
    This check protects numeric intent only, not every semantic qualifier.
    """
    candidate = candidate.strip().strip('"\'` \n')
    if not candidate or len(candidate) > 350:
        return original
    def normalized_num_tokens(s: str) -> set[str]:
        return {re.sub(r"[\s\u00a0]", "", n) for n in _question_numbers(s)}
    if not normalized_num_tokens(original).issubset(normalized_num_tokens(candidate)):
        return original
    return original if normalize(candidate) == normalize(original) else original + "\n" + candidate


def rewrite_with_ollama(question: str) -> str:
    """Cheap conditional query expansion; does NOT answer user questions."""
    from ollama import Client
    from src.config import OLLAMA_BASE_URL
    try:
        response = Client(host=OLLAMA_BASE_URL).chat(
            model="qwen2.5:3b",
            stream=False,
            messages=[
                {"role": "system", "content": (
                    "Magyar banki dokumentumok szemantikus kereséséhez írj egyetlen "
                    "rövid alternatív keresőkifejezést. Ne válaszold meg a kérdést. "
                    "A forintösszegeket, limithatárokat, pénznemet, csatornát és "
                    "tranzakciótípust változtatás nélkül őrizd meg. "
                    "Csak az új keresőkifejezést írd ki."
                )},
                {"role": "user", "content": question},
            ],
            options={"temperature": 0, "num_predict": 90, "num_ctx": 1024},
        )
        message = response.get("message") if isinstance(response, dict) else getattr(response, "message", None)
        candidate = message.get("content", "") if isinstance(message, dict) else getattr(message, "content", "")
        return safe_search_rewrite(question, candidate)
    except Exception:
        return question


def conflicting_section_payloads(*groups: list[Hit]) -> list[str]:
    """Fail closed for two different payloads claiming the SAME source version.

    The vector and structured representations can have different wrappers,
    so only compare payloads with the same retrieval method. This does NOT
    detect general cross-document legal contradictions.
    """
    seen: dict[tuple[str, str, str, str], str] = {}
    conflicts: set[str] = set()
    for group in groups:
        for hit in group:
            meta = hit.get("metadata", {})
            method = str(hit.get("retrieval_method", "injected"))
            key = (str(meta.get("doc_id", "")), str(meta.get("section_id", "")),
                   str(meta.get("effective_from", "")), method)
            if not key[0] or not key[1]:
                continue
            normalized = " ".join(str(hit.get("content", "")).split())
            if key in seen and seen[key] != normalized:
                conflicts.add(f"{key[0]}:{key[1]}")
            seen[key] = normalized
    return sorted(conflicts)


def merge_hits(first: list[Hit], second: list[Hit], limit: int = 10) -> list[Hit]:
    """Deduplicate by original document and section; keep first occurrence."""
    merged: list[Hit] = []
    seen: set[tuple[str, str]] = set()
    for h in first + second:
        meta = h.get("metadata", {})
        key = (str(meta.get("doc_id", "")), str(meta.get("section_id", "")))
        if key in seen:
            continue
        seen.add(key)
        merged.append(h)
        if len(merged) >= limit:
            break
    return merged


def build_graph(
    retrieve_fn: RetrieveFn | None = None,
    generate_fn: GenerateFn | None = None,
    rewrite_fn: RewriteFn | None = None,
    targeted_fn: Callable[..., list[Hit]] | None = None,
):
    """Compile a LangGraph with a maximum of one retrieval rewrite.

    Optional injected functions make the workflow testable without real Ollama.
    """
    from langgraph.graph import StateGraph, START, END
    from src.generation import answer_question
    from src.cross_document import plan_cross_document

    injected_retrieval = retrieve_fn is not None
    if retrieve_fn is None:
        from src.retrieval import retrieve_documents
        retrieve_fn = retrieve_documents
    generate_fn = generate_fn or answer_question
    rewrite_fn = rewrite_fn or rewrite_with_ollama

    def scope_gate(state: RAGState) -> dict[str, Any]:
        plan = plan_query(state["question"])
        if plan.scope != "in_scope":
            return {"scope": plan.scope, "result": scope_result(plan),
                    "steps": state.get("steps", []) + ["scope_gate:rejected"]}
        cross = plan_cross_document(state["question"])
        label = f"plan:{cross.kind}" if cross else "plan:standard"
        return {"scope": "in_scope", "steps": state.get("steps", []) + ["scope_gate:accepted", label]}

    def after_scope(state: RAGState) -> str:
        return END if state.get("scope") != "in_scope" else "retrieve"

    def retrieve(state: RAGState) -> dict[str, Any]:
        original = state["question"]
        query = state.get("search_query", original)
        rounds = state.get("retrieval_round", 0) + 1
        filtered = "glossary" if is_definition(original) else None
        if injected_retrieval:
            new_hits = retrieve_fn(query, k=TOP_K, document_type=filtered)
        else:
            new_hits = retrieve_routed(query, plan_query(original), k=TOP_K, vector_fn=retrieve_fn)
        # When the initial query needed a rewrite, prioritize new hits while
        # retaining the original ones as additional candidates.
        collisions = conflicting_section_payloads(new_hits, state.get("hits", []))
        combined = merge_hits(new_hits, state.get("hits", []), limit=16)
        return {
            "conflict_ids": sorted(set(state.get("conflict_ids", [])) | set(collisions)),
            "hits": combined,
            "retrieval_round": rounds,
            "steps": state.get("steps", []) + [f"retrieve:{rounds}"],
        }

    def grade(state: RAGState) -> dict[str, Any]:
        if state.get("conflict_ids"):
            return {"relevance": "conflict", "relevance_reason": "Azonos forrásszakasz eltérő szövegváltozatai: " + ", ".join(state["conflict_ids"]),
                    "steps": state.get("steps", []) + ["grade:conflict"]}
        good, reason = grade_retrieval(state["question"], state.get("hits", []))
        missing = []
        from src.faceted_qa import plan_facets, missing_facets
        from src.cross_document import plan_cross_document
        from src.legal_reasoning import plan_legal
        facet_plan = plan_facets(state["question"])
        if facet_plan and not plan_cross_document(state["question"]) and not plan_legal(state["question"]):
            missing = list(missing_facets(facet_plan, state.get("hits", [])))
            if missing:
                good = False
                reason = f"Hiányzó alátámasztott kérdésrészek: {len(missing)}/{len(facet_plan.facets)}"
        return {
            "missing_facets": missing,
            "relevance": "sufficient" if good else "weak",
            "relevance_reason": reason,
            "steps": state.get("steps", []) + ["grade:" + ("sufficient" if good else "weak")],
        }

    def after_grade(state: RAGState) -> str:
        from src.legal_reasoning import plan_legal
        if state.get("relevance") == "conflict":
            return "answer"
        if state.get("relevance") == "weak" and (
            plan_legal(state["question"]) or plan_cross_document(state["question"])
        ):
            return "targeted_legal" if not state.get("targeted_legal_done") else "answer"
        if state.get("missing_facets"):
            return "answer" if state.get("targeted_facets_done") else "targeted_facets"
        if state.get("relevance") == "weak" and state.get("rewrite_count", 0) < MAX_REWRITES:
            return "rewrite"
        return "answer"

    def targeted_legal(state: RAGState) -> dict[str, Any]:
        from src.legal_reasoning import plan_legal, targeted_legal_hits
        from src.cross_document import targeted_cross_hits
        cross = plan_cross_document(state["question"])
        legal = plan_legal(state["question"]) if not cross else None
        plan = cross or legal
        if targeted_fn:
            retrieved = targeted_fn(plan, state.get("hits", [])) if plan else []
        elif cross:
            retrieved = targeted_cross_hits(cross, state.get("hits", []))
        else:
            retrieved = targeted_legal_hits(legal, state.get("hits", [])) if legal else []
        collision = conflicting_section_payloads(state.get("hits", []), retrieved)
        combined = merge_hits(state.get("hits", []), retrieved, limit=16)
        rounds = state.get("retrieval_round", 0) + (1 if retrieved else 0)
        label = "targeted_multidoc" if cross else "targeted_legal"
        return {
            "hits": combined,
            "conflict_ids": sorted(set(state.get("conflict_ids", [])) | set(collision)),
            "targeted_legal_done": True,
            "retrieval_round": rounds,
            "steps": state.get("steps", []) + [
                f"{label}:{len(retrieved)}",
                f"retrieve:{rounds}" if retrieved else f"{label}:no_new_sections",
            ],
        }

    def targeted_facets(state: RAGState) -> dict[str, Any]:
        """One bounded per-facet retrieval; no hardcoded section IDs.

        Unlike deterministic legal target lookups, this uses the same routed
        vector retrieval available to Basic. It only queries the subquestions
        for which the initial evidence did not support an answer.
        """
        from src.faceted_qa import plan_facets
        facet_plan = plan_facets(state["question"])
        wanted = set(state.get("missing_facets", []))
        retrieved: list[Hit] = []
        if facet_plan:
            for facet in facet_plan.facets:
                if facet not in wanted:
                    continue
                if injected_retrieval:
                    sub_hits = retrieve_fn(facet, k=TOP_K, document_type=None)
                else:
                    sub_hits = retrieve_routed(facet, plan_query(facet), k=TOP_K, vector_fn=retrieve_fn)
                retrieved.extend(sub_hits)
        collisions = conflicting_section_payloads(state.get("hits", []), retrieved)
        combined = merge_hits(state.get("hits", []), retrieved, limit=24)
        rounds = state.get("retrieval_round", 0) + (1 if retrieved else 0)
        return {
            "hits": combined,
            "conflict_ids": sorted(set(state.get("conflict_ids", [])) | set(collisions)),
            "targeted_facets_done": True,
            "retrieval_round": rounds,
            "steps": state.get("steps", []) + [
                f"targeted_facets:{len(wanted)}",
                f"retrieve:{rounds}" if retrieved else "targeted_facets:no_new_sections",
            ],
        }

    def rewrite(state: RAGState) -> dict[str, Any]:
        expanded = rewrite_fn(state["question"])
        # Safety: injected rewrite functions also have to obey invariants.
        search_query = safe_search_rewrite(state["question"], expanded)
        return {
            "search_query": search_query,
            "rewrite_count": state.get("rewrite_count", 0) + 1,
            "steps": state.get("steps", []) + ["rewrite"],
        }

    def answer(state: RAGState) -> dict[str, Any]:
        if state.get("conflict_ids"):
            result = {"status": "insufficient", "sources": [], "evidence": [],
                      "answer": "Ellentmondó változatok találhatók ugyanahhoz a forrásszakaszhoz. Ellenőrzés nélkül nem adható megbízható válasz.",
                      "mode": "duplicate_source_conflict", "conflicting_sections": state["conflict_ids"]}
        else:
            # Shared answer engine for Basic and Agentic; the graph's advantage can only come from gathering missing evidence, not different facts.
            result = generate_fn(state["question"], hits_override=state.get("hits", []))
        return {"result": result, "steps": state.get("steps", []) + ["answer:" + result.get("status", "unknown")]}

    def after_answer(state: RAGState) -> str:
        # Answer validation failure is not evidence of retrieval failure.
        # An LLM rewrite here caused 30+ second tails and repeated false refusals.
        return END

    builder = StateGraph(RAGState)
    builder.add_node("scope_gate", scope_gate)
    builder.add_node("retrieve", retrieve)
    builder.add_node("grade", grade)
    builder.add_node("targeted_legal", targeted_legal)
    builder.add_node("targeted_facets", targeted_facets)
    builder.add_node("rewrite", rewrite)
    builder.add_node("answer", answer)
    builder.add_edge(START, "scope_gate")
    builder.add_conditional_edges("scope_gate", after_scope, {"retrieve": "retrieve", END: END})
    builder.add_edge("retrieve", "grade")
    builder.add_conditional_edges("grade", after_grade, {
        "targeted_legal": "targeted_legal", "targeted_facets": "targeted_facets", "rewrite": "rewrite", "answer": "answer"
    })
    builder.add_edge("targeted_legal", "grade")
    builder.add_edge("targeted_facets", "grade")
    builder.add_edge("rewrite", "retrieve")
    builder.add_conditional_edges("answer", after_answer, {"rewrite": "rewrite", END: END})
    return builder.compile()


def ask_agentic(question: str) -> dict[str, Any]:
    if not question.strip():
        raise ValueError("Question must not be empty")
    plan = plan_query(question)
    if plan.scope != "in_scope":
        result = scope_result(plan)
        result.update({"graph_trace": ["scope_gate:rejected"],
                       "retrieval_rounds": 0, "rewrite_count": 0,
                       "retrieval_grade": "out_of_scope",
                       "retrieval_grade_reason": plan.reason})
        return result
    state = build_graph().invoke({
        "question": question.strip(),
        "search_query": question.strip(),
        "retrieval_round": 0,
        "rewrite_count": 0,
        "steps": [],
        "targeted_legal_done": False,
        "targeted_facets_done": False,
    }, config={"recursion_limit": 12})
    result = dict(state.get("result", {}))
    result["graph_trace"] = state.get("steps", [])
    result["retrieval_rounds"] = state.get("retrieval_round", 0)
    result["rewrite_count"] = state.get("rewrite_count", 0)
    result["retrieval_grade"] = state.get("relevance")
    result["retrieval_grade_reason"] = state.get("relevance_reason")
    return result
