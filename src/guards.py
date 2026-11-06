"""
Small, explicit domain guard for an amount-limited source tariff.
This is NOT a universal bank-fee reasoning engine; it only recognizes a
clearly documented own-ATM withdrawal ceiling in a retrieved Markdown table.
"""
import re
from typing import Any

AMOUNT_RE = re.compile(r"\b(\d{1,3}(?:[\s\u00a0]\d{3})+|\d+)\s*Ft\b", re.I)
CAP_RE = re.compile(r"\b(\d{1,3}(?:[\s\u00a0]\d{3})+|\d+)\s*Ft\s+tranzakcióösszegig\b", re.I)


def _int_amount(number: str) -> int:
    return int(re.sub(r"[\s\u00a0]", "", number))


def check_documented_atm_ceiling(question: str, hits: list[dict[str, Any]]) -> dict[str, Any] | None:
    """Refuse extrapolating an explicitly capped *domestic own-ATM* tariff.

    Only activates when question explicitly names 'saját ATM' and supplies a
    forint amount. The returned source is the original table section.
    """
    if not re.search(r"\bsaját\s+ATM\b", question, re.IGNORECASE):
        return None
    match = AMOUNT_RE.search(question)
    if not match:
        return None
    amount = _int_amount(match.group(1))
    for hit in hits:
        for line in hit["content"].splitlines():
            if "|" not in line or not re.search(r"saját\s+ATM-ből", line, re.IGNORECASE):
                continue
            cap_match = CAP_RE.search(line)
            if not cap_match:
                continue
            cap = _int_amount(cap_match.group(1))
            if amount > cap:
                return {
                    "status": "insufficient",
                    "answer": (
                        f"A megtalált díjtáblázat a belföldi saját ATM-es készpénzfelvétel díját "
                        f"csak {cap:,} Ft tranzakcióösszegig adja meg. "
                        f"A kérdezett {amount:,} Ft-os összegre ebből a forrásból "
                        "nem állapítható meg megbízható díj."
                    ).replace(",", " "),
                    "sources": [
                        {"citation_id": "S1", **hit["metadata"]}
                    ],
                    "evidence": [
                        {"source_id": "S1", "quote": line.strip()}
                    ],
                    "guard": "atm_amount_exceeds_documented_limit",
                }
    return None


def check_paper_copy_fee_claim(question: str, answer: str, hits: list[dict[str, Any]]) -> None:
    """Reject a narrowly detectable wrong claim about the documented paper-copy fee.

    This is a domain-specific cross-check, not comprehensive semantic validation.
    Raises ValueError so the generator can retry/abstain.
    """
    q = question.casefold()
    if "másolat" not in q or "kivonat" not in q:
        return
    # Only enforce if a retrieved section explicitly quotes the nonzero copy fee.
    copy_fee_documented = any(
        re.search(r"bankszámlakivonat másolata papíralapon:\s*1\s*000\s*Ft", h["content"], re.I)
        for h in hits
    )
    if not copy_fee_documented:
        return
    a = answer.casefold()
    unsupported_free_copy = (
        re.search(r"másolat\w*\s+(?:is\s+|szintén\s+)?(?:0\s*ft|díjmentes)", a)
        or re.search(r"mindkettő\s+(?:0\s*ft|díjmentes)", a)
    )
    if unsupported_free_copy:
        raise ValueError("Answer incorrectly treats a 1 000 Ft statement copy as free")
