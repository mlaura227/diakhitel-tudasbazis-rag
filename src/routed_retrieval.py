"""
Intent-aware retrieval: source-backed sections + vector search.
Structured lookups are prefiltered by section ID; open-ended topics still use Chroma.
The returned hit shape is identical to src.retrieval.retrieve_documents.
"""
from __future__ import annotations
import re
from src.query_policy import QueryPlan
from src.source_sections import lookup_sections


def combine(*groups: list[dict], limit: int = 14) -> list[dict]:
    out=[];seen=set()
    for group in groups:
        for h in group:
            m=h.get('metadata',{})
            key=(m.get('doc_id'),m.get('section_id'))
            if key in seen: continue
            seen.add(key);out.append(h)
            if len(out)>=limit:return out
    return out


def retrieve_routed(question: str, plan: QueryPlan, k: int = 5, *, vector_fn=None) -> list[dict]:
    local=lookup_sections(plan.sections)
    if vector_fn is None:
        from src.retrieval import retrieve_documents
        vector_fn=retrieve_documents
    # Keep semantics of the existing vector retrieval, but don't let a glossary definition answer a monetary fee question.
    vector=[]
    # Add regulation evidence when the user asks about contractual rights, eligibility, opening/closing or exceptions. General regulation is not a substitute for a product-specific fee table.
    needs_rules = bool(re.search(
        r"jogosult|jogosults|felt[eé]tel|szerz[oő]d[eé]s|nyit[aá]s|"
        r"megnyit|nyithat|nyitni|felmond|megsz[uű]n|lej[aá]r|[aá]tszerz|"
        r"nagykor|kiskor|di[aá]khitel k[oö]zpont|ig[eé]nyl[eé]s|"
        r"[uü]zletszab[aá]lyzat",
        question.casefold()
    ))
    try:
        if plan.intents == ('definition',):
            vector=vector_fn(question,k=k,document_type='glossary')
            if needs_rules:
                vector=combine(vector,vector_fn(question,k=k,document_type='account_regulation'))
        elif plan.intents == ('fee',):
            # Supports Chroma filter {"document_type": {"$in": ...}}.
            vector=vector_fn(question,k=k,document_types=('fee_schedule','product_notice'))
            if needs_rules:
                vector=combine(vector,vector_fn(question,k=k,document_type='account_regulation'))
        elif 'definition' in plan.intents and 'fee' in plan.intents:
            vector=combine(
                vector_fn(question,k=k,document_type='glossary'),
                vector_fn(question,k=k,document_types=('fee_schedule','product_notice')),
                vector_fn(question,k=k,document_type='account_regulation') if needs_rules else [],
            )
        elif plan.intents == ('eligibility',) or needs_rules:
            vector=combine(
                vector_fn(question,k=k,document_type='account_regulation'),
                vector_fn(question,k=k,document_type='product_notice'),
                vector_fn(question,k=k,document_type='fee_schedule'),
            )
        else:
            vector=vector_fn(question,k=k)
    except TypeError:
        # Legacy monkeypatched/injected retriever that only takes document_type.
        vector=vector_fn(question,k=k)
    return combine(local,vector)
