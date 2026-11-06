"""
Exact local Markdown-section lookup (structured companion to Chroma search).
Never synthesizes tariffs.
Section contents and fee amounts come from the -rag.md files in data/processed.
No database rebuild is needed!
"""
from __future__ import annotations

from functools import lru_cache
import re
from src.config import SOURCES, resolve_source_path

FEE_ID = re.compile(r'^###\s+\[(SRC-P\d{2}-B\d{2})\]\s+(.+)$')
NOTICE_ID = re.compile(r'^##\s+(.+?)\s+\{#(source-[\w-]+)\}\s*$')
GLOSS_ID = re.compile(r'^###\s+(\d+)\.\s+(.+)$')
PAGE = re.compile(r'^(\d+)\. oldal')
CITE_PAGE = re.compile(r'\[Forrás:\s*(\d+(?:\s*[-–]\s*\d+)?)\. oldal')


@lru_cache(maxsize=1)
def catalog() -> dict[tuple[str, str], dict]:
    records = {}
    for spec in SOURCES:
        if spec.document_type not in ("fee_schedule", "product_notice", "glossary"):
            continue  # Legal knowledge is indexed in Chroma, not parsed as a fee table.
        path = resolve_source_path(spec.rag_filename)
        lines = path.read_text(encoding='utf-8-sig').splitlines()
        current = None
        page = 'unknown'
        parent = ''
        skip = False
        entries = []
        for line in lines + ['## __END__']:
            if line.startswith('## '):
                skip = 'forráshűségi' in line.casefold() or line == '## __END__'
                m = PAGE.match(line[3:])
                if m: page = m.group(1)
                parent = line
            found = None
            if spec.document_type == 'fee_schedule':
                m = FEE_ID.match(line)
                if m: found = (m.group(1),m.group(2),page)
            elif spec.document_type == 'product_notice':
                m = NOTICE_ID.match(line)
                if m: found = (m.group(2),m.group(1),'unknown')
            elif spec.document_type == 'glossary':
                m = GLOSS_ID.match(line)
                if m: found = (f'section-{int(m.group(1))+1:03d}',line[4:],'unknown')
            if current is not None and (found is not None or line.startswith('## ')):
                section_id,title,section_page,body = current
                content = '\n'.join(body).strip()
                if section_page == 'unknown':
                    match = CITE_PAGE.search(content)
                    if match: section_page = match.group(1).replace(' ','')
                meta = {
                    'doc_id': spec.doc_id, 'section_id': section_id, 'section_title': title,
                    'document_title':spec.document_title, 'document_type':spec.document_type,
                    'page':section_page, 'source':str(path),
                    'source_file':spec.rag_filename,
                    'document_date':spec.document_date, 'effective_from':spec.effective_from,
                    'publication_date':spec.publication_date,
                }
                records[(spec.doc_id,section_id)] = {'content':content, 'metadata':meta, 'distance':0.0, 'retrieval_method':'structured_section'}
                current = None
            if found is not None and not skip:
                sid,title,spage=found
                current=(sid,title,spage,[line])
            elif current is not None:
                current[3].append(line)
    # The fee schedule's introductory applicability paragraph is a level-2 heading, not a SRC-* row. Preserve it as a synthetic section so the date cannot be confused with the product notice's effective date.
    fee = next((s for s in SOURCES if s.doc_id == 'fee_schedule_20260901'), None)
    if fee:
        text=resolve_source_path(fee.rag_filename).read_text(encoding='utf-8-sig')
        match=re.search(r'(?ms)^## Dokumentumazonosítás és hatály\s*\n(.*?)(?=^## |\Z)',text)
        if match:
            records[(fee.doc_id, 'section-001')] = {
                'content': '## Dokumentumazonosítás és hatály\n' + match.group(1).strip(),
                'metadata': {'doc_id': fee.doc_id, 'section_id': 'section-001',
                             'section_title': 'Dokumentumazonosítás és hatály',
                             'document_title':fee.document_title,
                             'document_type':fee.document_type,'page':'unknown',
                             'source':str(resolve_source_path(fee.rag_filename)),
                             'source_file':fee.rag_filename,
                             'document_date':fee.document_date,
                             'effective_from':fee.effective_from,
                             'publication_date':fee.publication_date},
                'distance': 0.0, 'retrieval_method': 'structured_section',
            }
    return records


def lookup_sections(keys: tuple[tuple[str,str],...] | list[tuple[str,str]]) -> list[dict]:
    items=catalog()
    return [items[k] for k in keys if k in items]
