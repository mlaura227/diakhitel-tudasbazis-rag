"""
Evidence-first, source-derived responses for unambiguous bank facts.
Values are parsed from retrieved text, never embedded as fee constants.
"""
from __future__ import annotations

import re
import unicodedata
from typing import Any

Hit = dict[str, Any]


def norm(value: str) -> str:
    """Accent-insensitive matching for Hungarian topic routing."""
    normalized = unicodedata.normalize("NFKD", value.casefold())
    return "".join(c for c in normalized if not unicodedata.combining(c))


def find_section(hits: list[Hit], doc_id: str, section_id: str) -> Hit | None:
    return next(
        (h for h in hits if h["metadata"].get("doc_id") == doc_id
         and h["metadata"].get("section_id") == section_id), None
    )


def find_line(hit: Hit | None, *needles: str) -> str | None:
    if hit is None:
        return None
    for line in hit["content"].splitlines():
        if all(norm(needle) in norm(line) for needle in needles):
            return line.strip()
    return None


def amount_from_line(line: str, pattern: str) -> str | None:
    match = re.search(pattern, line, flags=re.IGNORECASE)
    return match.group(1).strip() if match else None


def response(answer: str, quotations: list[tuple[Hit, str]]) -> dict[str, Any]:
    """Attach source IDs to exact, source-derived quotations."""
    sources, evidence = [], []
    seen: dict[tuple[str, str], str] = {}
    for hit, quote in quotations:
        if quote not in hit["content"]:
            raise ValueError("Extractive quote is absent from its source")
        meta = hit["metadata"]
        key = (str(meta.get("doc_id")), str(meta.get("section_id")))
        if key not in seen:
            sid = f"S{len(seen) + 1}"
            seen[key] = sid
            sources.append({"citation_id": sid, **meta})
        evidence.append({"source_id": seen[key], "quote": quote})
    return {
        "status": "answered", "answer": answer, "sources": sources,
        "evidence": evidence, "mode": "extractive", "attempts": 0,
    }


def _maintenance(hits: list[Hit]) -> tuple[Hit, str, str, str, str] | None:
    h = find_section(hits, "fee_schedule_20260901", "SRC-P03-B01")
    electronic = find_line(h, "Elektronikus kivonattal:", "havi díj")
    paper = find_line(h, "Papíralapú kivonattal:", "havi díj")
    if not h or not electronic or not paper:
        return None
    monthly_e = amount_from_line(electronic, r"havi díj\s+([\d\s]+\s*Ft)")
    annual_e = amount_from_line(electronic, r"teljes éves díj\s+([\d\s]+\s*Ft)")
    monthly_p = amount_from_line(paper, r"havi díj\s+([\d\s]+\s*Ft)")
    annual_p = amount_from_line(paper, r"teljes éves díj\s+([\d\s]+\s*Ft)")
    if not all((monthly_e, annual_e, monthly_p, annual_p)):
        return None
    if monthly_e != monthly_p or annual_e != annual_p:
        # Don't compress different tariffs into an "equal fee" claim.
        return None
    return h, electronic, paper, monthly_e, annual_e


def _statement_copy(hits: list[Hit]) -> tuple[Hit, str, str] | None:
    h = find_section(hits, "fee_schedule_20260901", "SRC-P09-B03")
    quote = find_line(h, "bankszámlakivonat másolata papíralapon:")
    if not h or not quote:
        return None
    fee = amount_from_line(quote, r"másolata papíralapon:\s*([\d\s]+\s*Ft)")
    return (h, quote, fee) if fee else None


def _definition(question: str, hits: list[Hit]) -> dict[str, Any] | None:
    """Return the exact relevant glossary definition rather than generating one."""
    q = norm(question)
    if not any(phrase in q for phrase in ("mit jelent", "mi az a ", "mi a ", "fogalma", "definicio", "definiald")):
        return None
    stop = {"mit", "jelent", "hogyan", "mi", "az", "egy", "van", "ennek", "fogalma", "definicio", "kerlek"}
    q_terms = {x for x in re.findall(r"[a-z0-9]+", q) if len(x) >= 4 and x not in stop}
    if not q_terms:
        return None

    def lexical_score(hit: Hit) -> tuple[int, float]:
        if hit["metadata"].get("document_type") != "glossary":
            return (0, 0.0)
        title = norm(str(hit["metadata"].get("section_title", "")))
        title_terms = {x for x in re.findall(r"[a-z0-9]+", title) if len(x) >= 4}
        found = sum(any(t == w or (len(t) >= 6 and w.startswith(t[:6])) for w in title_terms)
                    for t in q_terms)
        return found, found / max(1, len(title_terms))

    ranked = sorted(hits, key=lexical_score, reverse=True)
    if not ranked or lexical_score(ranked[0])[0] == 0:
        return None
    best = ranked[0]
    # Crucial: an explicit acronym in the question must occur in the title.
    # "SEPA" must never resolve to "Rendszeres átutalás".
    acronyms = re.findall(r"\b[A-ZÁÉÍÓÖŐÚÜŰ]{3,}\b", question)
    title_norm = norm(str(best["metadata"].get("section_title", "")))
    if any(norm(a) not in title_norm for a in acronyms):
        return None
    for line in best["content"].splitlines():
        candidate = line.strip()
        if not candidate or candidate.startswith("#") or candidate.startswith("Forrásdokumentum:"):
            continue
        quote = re.sub(r"\s+\[Forrás:.*$", "", candidate).strip()
        if len(quote) < 24 or quote not in best["content"]:
            continue
        title = re.sub(r"^\d+\.\s*", "", str(best["metadata"].get("section_title", "")))
        return response(f'A glosszárium „{title}” meghatározása: {quote}', [(best, quote)])
    return None


def extract_answer(question: str, hits: list[Hit]) -> dict[str, Any] | None:
    """Answer supported intents; return None for questions needing broader reasoning."""
    q = norm(question)
    wants_copy = "kivonat" in q and ("masolat" in q or "utolag" in q)
    wants_regular = ("papir" in q or "postaz" in q or "rendszeres" in q)
    maintenance = _maintenance(hits)
    copy = _statement_copy(hits)

    if wants_copy and copy:
        copy_hit, copy_quote, copy_fee = copy
        if wants_regular and maintenance:
            acc_hit, el, paper, monthly, annual = maintenance
            return response(
                f"A papíralapú kivonattal vezetett OTP Diákhitel Számla "
                f"számlavezetési díja havonta {monthly}, évente {annual}. "
                f"A postai úton kiküldött rendszeres havi bankszámlakivonat "
                f"papíralapú másolatának díja {copy_fee}.",
                [(acc_hit, paper), (copy_hit, copy_quote)],
            )
        if not wants_regular:
            return response(
                f"A postai úton kiküldött rendszeres havi bankszámlakivonat "
                f"papíralapú másolatának díja {copy_fee}.",
                [(copy_hit, copy_quote)],
            )

    if ("szamlavezet" in q or "szamladij" in q) and maintenance and not wants_copy:
        acc_hit, el, paper, monthly, annual = maintenance
        return response(
            f"Az OTP Diákhitel Számla számlavezetési díja elektronikus és "
            f"papíralapú kivonat esetén is havonta {monthly}, évente {annual}.",
            [(acc_hit, el), (acc_hit, paper)],
        )

    if "kartya" in q and any(t in q for t in ("potlas", "potkartya", "potkartya", "potol")):
        h = find_section(hits, "notice_20251118", "source-p1-card-fees")
        quote = find_line(h, "Mastercard Online Diákhitel pótkártya díja")
        if quote and h:
            cells = [cell.strip() for cell in quote.strip("|").split("|")]
            if len(cells) == 2 and cells[1] and cells[1] not in {"---", "-"}:
                return response(
                    f"A Mastercard Online Diákhitel pótkártya díja {cells[1]}.",
                    [(h, quote)],
                )

    return _definition(question, hits)
