"""Test retrieval quality and source attribution, without LLM generation."""
from src.retrieval import retrieve_documents

QUESTIONS = [
    "Mennyibe kerül az OTP Diákhitel Számla számlavezetése?",
    "Ingyenes a papíralapú rendszeres számlakivonat vagy az utólag kért másolat?",
    "Mennyibe kerül a Mastercard Online Diákhitel kártya pótlása?",
    "Mikortól hatályos a Diákhitel Számla hirdetménye?",
    "Milyen díja van a devizajóváírásnak?",
    "Mit jelent a SEPA átutalás?",
    "Mennyibe kerül az ATM készpénzfelvétel?",
]

if __name__ == "__main__":
    for question in QUESTIONS:
        print("\n" + "=" * 80 + f"\nQUESTION: {question}")
        for rank, result in enumerate(retrieve_documents(question, k=5), start=1):
            m = result["metadata"]
            print(
                f"\nRESULT {rank} | distance={result['distance']:.4f} | "
                f"source={m['doc_id']} | page={m['page']} | "
                f"section={m['section_id']}"
            )
            print("TITLE:", m["section_title"])
            print(result["content"][:450].replace("\n", " "))
