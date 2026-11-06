"""Traceable Chroma retrieval with optional single/multiple metadata types."""
from src.vector_store import get_vector_store


def retrieve_documents(question: str, k: int = 5,
                       document_type: str | None = None,
                       document_types: tuple[str, ...] | None = None) -> list[dict]:
    if document_type and document_types:
        raise ValueError('Choose either document_type or document_types')
    store=get_vector_store()
    kwargs={'k':k}
    if document_type:
        kwargs['filter']={'document_type':document_type}
    elif document_types:
        kwargs['filter']={'document_type':{'$in':list(document_types)}}
    docs=store.similarity_search_with_score(question, **kwargs)
    return [
        {'content':doc.page_content, 'metadata':doc.metadata,
         'distance':float(distance), 'retrieval_method':'vector'}
        for doc,distance in docs
    ]
