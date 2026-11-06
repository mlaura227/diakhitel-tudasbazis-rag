"""Read and section the four-document corpus without splitting fee rows or legal topics."""
from __future__ import annotations
import hashlib
import re
import unicodedata
from langchain_core.documents import Document
from src.config import SOURCES, SourceSpec, resolve_source_path

HEADING_RE = re.compile(r"^(#{2,3})\s+(.+)$")
FEE_PAGE_RE = re.compile(r"^(\d+)\. oldal\b", re.IGNORECASE)
CITED_PAGE_RE = re.compile(r"\[Forrás:\s*(\d+(?:\s*[-–]\s*\d+)?)\. oldal", re.IGNORECASE)
SRC_ID_RE = re.compile(r"\[(SRC-[A-Z0-9-]+)\]")
ANCHOR_RE = re.compile(r"\{#([^}]+)\}")
# E.g. '*Forrás: 1., 6–8. oldal, I.1.; hatály: ...*'
KNOWLEDGE_PAGES = re.compile(r"\*Forrás:\s*([^\n]*?)\s+oldal", re.IGNORECASE)


def _clean_heading(heading: str) -> str:
    return ANCHOR_RE.sub("", heading).strip()


def _is_metadata_heading(heading: str) -> bool:
    return any(t in heading.lower() for t in
               ("dokumentumazonosítás", "dokumentum metaadatai", "dokumentumadatok"))


def _is_note_heading(heading: str) -> bool:
    return "forráshűségi és értelmezési megjegyzések" in heading.lower()


def _metadata(spec: SourceSpec, source_path, page: str, section_id: str,
              title: str, section_num: int, *, source_pages: str = "") -> dict:
    return {
        "source": str(source_path),
        "source_file": source_path.name,
        "original_pdf": spec.original_pdf,
        "source_transcription_file": spec.rag_filename,
        "source_representation": spec.representation,
        "product_scope": spec.product_scope,
        "doc_id": spec.doc_id,
        "document_title": spec.document_title,
        "document_type": spec.document_type,
        "document_date": spec.document_date,
        "publication_date": spec.publication_date,
        "effective_from": spec.effective_from,
        "page": page or "unknown",
        "source_pages": source_pages or page or "unknown",
        "section_id": section_id,
        "section_title": title,
        "section_number": section_num,
    }


def _section_documents(text: str, spec: SourceSpec, source_path) -> list[Document]:
    """Preserve the existing 3-source heading strategy and its stable IDs."""
    docs: list[Document] = []
    parent_heading = ""
    page_from_heading = ""
    current_heading = ""
    current_lines: list[str] = []
    current_page = ""
    inside_notes = False

    def flush() -> None:
        nonlocal current_heading, current_lines, current_page
        if not current_heading:
            return
        content = "\n".join(current_lines).strip()
        if content:
            section_num = len(docs) + 1
            source_match = SRC_ID_RE.search(current_heading)
            anchor_match = ANCHOR_RE.search(current_heading)
            sid = (source_match.group(1) if source_match else
                   anchor_match.group(1) if anchor_match else f"section-{section_num:03d}")
            title = _clean_heading(current_heading)
            page = current_page
            if not page:
                m = CITED_PAGE_RE.search(content)
                if m: page = m.group(1).replace(" ", "")
            indexed_text = (
                f"Forrásdokumentum: {spec.document_title}\n"
                f"{parent_heading + chr(10) if parent_heading and spec.document_type != 'product_notice' else ''}"
                f"{content}"
            )
            docs.append(Document(
                page_content=indexed_text,
                metadata=_metadata(spec, source_path, page, sid, title, section_num),
            ))
        current_heading, current_lines, current_page = "", [], ""

    for line in text.splitlines():
        m = HEADING_RE.match(line)
        if not m:
            if current_heading and not inside_notes:
                current_lines.append(line)
            continue
        level, heading = len(m.group(1)), m.group(2)
        if level == 2:
            flush()
            inside_notes = _is_note_heading(heading)
            if inside_notes:
                parent_heading = ""
                continue
            parent_heading = f"## {_clean_heading(heading)}"
            if spec.document_type == "fee_schedule":
                p = FEE_PAGE_RE.match(heading)
                page_from_heading = p.group(1) if p else ""
            else:
                page_from_heading = ""
            if spec.document_type == "product_notice" or _is_metadata_heading(heading):
                current_heading, current_lines, current_page = heading, [line], page_from_heading
            continue
        if spec.document_type in ("fee_schedule", "glossary"):
            flush()
            if not inside_notes:
                current_heading, current_lines, current_page = heading, [line], page_from_heading
        elif current_heading and not inside_notes:
            current_lines.append(line)
    flush()
    return docs


def _regulation_documents(text: str, spec: SourceSpec, source_path) -> list[Document]:
    """Use each manually authored knowledge section as an intact retrieval chunk."""
    sections: list[tuple[str, list[str]]] = []
    heading: str | None = None
    lines: list[str] = []
    for line in text.splitlines():
        if line.startswith("## "):
            if heading is not None:
                sections.append((heading, lines))
            heading, lines = line[3:].strip(), [line]
        elif heading is not None:
            lines.append(line)
    if heading is not None:
        sections.append((heading, lines))

    docs = []
    seen: set[str] = set()
    for section_num, (title, block) in enumerate(sections, start=1):
        body = "\n".join(block).strip()
        if not body:
            continue
        if "*Forrás:" not in body:
            raise ValueError(f"Missing page/source citation in regulation section: {title}")
        m = KNOWLEDGE_PAGES.search(body)
        if not m:
            raise ValueError(f"Cannot identify source pages for regulation section: {title}")
        # Hash of the title is stable if other topics are inserted elsewhere.
        norm_title = unicodedata.normalize("NFC", title).casefold().strip()
        sid = "REG-" + hashlib.sha256(norm_title.encode("utf-8")).hexdigest()[:12].upper()
        if sid in seen:
            raise ValueError(f"Duplicate regulation title: {title}")
        seen.add(sid)
        page_string = m.group(1).strip(" ,.;")
        # Capture all source pages as written, without silently collapsing ranges.
        docs.append(Document(
            page_content=f"Forrásdokumentum: {spec.document_title}\n{body}",
            metadata=_metadata(spec, source_path, page_string, sid,
                               title, section_num, source_pages=page_string),
        ))
    if not docs:
        raise ValueError("No knowledge sections identified in the new regulation")
    return docs


def load_documents() -> list[Document]:
    """Load all four source specifications from supported processed-data folders."""
    docs = []
    for spec in SOURCES:
        path = resolve_source_path(spec.filename_to_index)
        if spec.document_type == "account_regulation":
            # The -rag.md is required as an original-page fidelity reference.
            resolve_source_path(spec.rag_filename)
            new = _regulation_documents(path.read_text(encoding="utf-8-sig"), spec, path)
        else:
            new = _section_documents(path.read_text(encoding="utf-8-sig"), spec, path)
        docs.extend(new)
    if not docs:
        raise ValueError("No sections found")
    keys = [(d.metadata["doc_id"], d.metadata["section_id"]) for d in docs]
    if len(set(keys)) != len(keys):
        raise ValueError("Duplicate document/section IDs in the corpus")
    return docs


def split_documents(documents: list[Document]) -> list[Document]:
    """Do not split these human-defined legal or tariff sections."""
    return documents


if __name__ == "__main__":
    from collections import Counter
    docs = load_documents()
    for doc_id, count in Counter(d.metadata["doc_id"] for d in docs).items():
        print(f"{doc_id}: {count} sections")
    print("Total:", len(docs))
    print("Regulation example:", next(d.metadata for d in docs if d.metadata["document_type"] == "account_regulation"))
