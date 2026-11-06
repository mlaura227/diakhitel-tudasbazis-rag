"""Corpus manifest and storage locations (OTP Diakhitel local RAG)."""
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
# Backwards compatible with the earlier project, plus the user's processed_data folder.
PROCESSED_DATA_DIR = PROJECT_ROOT / "data" / "processed"
PROCESSED_DATA_LOCATIONS = (
    PROCESSED_DATA_DIR,
    PROJECT_ROOT / "processed_data",
    PROJECT_ROOT / "data" / "processed_data",
)
CHROMA_PATH = PROJECT_ROOT / "chroma_db"
COLLECTION_NAME = "otp_diakhitel_regulation_v2"
EMBEDDING_MODEL = "bge-m3"
OLLAMA_BASE_URL = "http://localhost:11434"


@dataclass(frozen=True)
class SourceSpec:
    doc_id: str
    rag_filename: str
    document_title: str
    document_type: str
    original_pdf: str
    document_date: str = ""
    effective_from: str = ""
    publication_date: str = ""
    indexed_filename: str = ""  # If blank, index the source-faithful -rag.md
    representation: str = "source_transcription"
    product_scope: str = "otp_diakhitel"

    @property
    def filename_to_index(self) -> str:
        return self.indexed_filename or self.rag_filename


def resolve_source_path(filename: str) -> Path:
    """Find a file in either supported processed-data layout.

    Prevent silently choosing between conflicting copies of the same filename.
    """
    candidates = [folder / filename for folder in PROCESSED_DATA_LOCATIONS]
    found = [p for p in candidates if p.is_file()]
    if not found:
        tried = "\n".join(f"  - {p}" for p in candidates)
        raise FileNotFoundError(f"Missing processed source: {filename}\nSearched:\n{tried}")
    if len(found) > 1:
        import hashlib
        checksums = {hashlib.sha256(p.read_bytes()).hexdigest() for p in found}
        if len(checksums) > 1:
            raise ValueError(
                f"Conflicting copies of {filename}:\n" +
                "\n".join(f"  - {p}" for p in found) +
                "\nPlease keep only one version or make their contents identical."
            )
    return found[0]


SOURCES: tuple[SourceSpec, ...] = (
    SourceSpec(
        doc_id="fee_schedule_20260901",
        rag_filename="Dijjegyzek_OTP_Diakhitel_szamla_20260901-rag.md",
        document_title="OTP Diákhitel Számla – Díjjegyzék",
        document_type="fee_schedule",
        original_pdf="Dijjegyzek_OTP_Diakhitel_szamla_20260901.pdf",
        document_date="2026-09-01",  # NOT independently confirmed as effective date
    ),
    SourceSpec(
        doc_id="notice_20251118",
        rag_filename="Diakhitel_Szamla_H_20251118-rag.md",
        document_title="Hirdetmény az OTP Diákhitel Számláról",
        document_type="product_notice",
        original_pdf="Diakhitel_Szamla_H_20251118.pdf",
        publication_date="2025-09-18",
        effective_from="2025-11-18",
    ),
    SourceSpec(
        doc_id="glossary_undated",
        rag_filename="Glosszarium-rag.md",
        document_title="Glosszárium",
        document_type="glossary",
        original_pdf="Glosszarium.pdf",
        product_scope="general_banking_definitions",
    ),
    SourceSpec(
        doc_id="account_rules_20261015",
        rag_filename="USZ_LAK_Bankszamla_20261015-rag.md",
        indexed_filename="USZ_LAK_Bankszamla_20261015-knowledge.md",
        document_title="Üzletszabályzat – Fizetési számlák (Bankszámlák)",
        document_type="account_regulation",
        original_pdf="USZ_LAK_Bankszamla_20261015.pdf",
        document_date="2026-10-15",
        publication_date="2026-10-15",
        effective_from="2026-10-15",
        representation="interpreted_knowledge",
        product_scope="multiple_otp_retail_accounts",
    ),
)
