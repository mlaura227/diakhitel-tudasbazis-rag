"""Live Chroma retrieval probes; should be run after a successful ingestion."""
from src.config import COLLECTION_NAME
from src.retrieval import retrieve_documents
from src.vector_store import get_vector_store


def main():
    store = get_vector_store()
    counts = store._collection.count()
    print(f"Collection: {COLLECTION_NAME} | sections: {counts}")
    if counts != 133:
        raise RuntimeError(f"Expected 133 sections, got {counts}. Check ingestion logs.")

    probes = [
        ('Diákhitel szabályok',
         'Meddig lehet OTP Diákhitel Számlát nyitni, és mikor szűnik meg a szerződés?',
         'account_regulation'),
        ('Általános bankszámlaszerződés',
         'Mi a sorrend az egyedi számlaszerződés, a hirdetmény és az üzletszabályzat között?',
         'account_regulation'),
        ('Diákhitel termékhirdetmény',
         'Mikor lépett hatályba a Diákhitel Számla hirdetménye?',
         'product_notice'),
        ('Régi díjsor megőrzése',
         'Mennyi a Diákhitel Számla havi számlavezetési díja?',
         'fee_schedule'),
    ]
    for name, query, doc_type in probes:
        print(f"\n### {name}\nQUESTION: {query}")
        results = retrieve_documents(query, k=3, document_type=doc_type)
        if not results:
            print('NO RESULTS');continue
        for rank, h in enumerate(results, 1):
            m = h['metadata']
            print(f"{rank}. dist={h['distance']:.4f} | {m['doc_id']} | "
                  f"{m['section_title']} | page={m.get('source_pages',m.get('page'))}")
            print('   ', h['content'].replace('\n', ' ')[:240])


if __name__ == '__main__':
    main()
