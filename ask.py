"""Simple interactive terminal for the evidence-verified basic RAG baseline."""
from src.generation import answer_question


def main() -> None:
    print("OTP Diákhitel Számla - ellenőrzött forrásjelölt alap RAG (exit: kilépés)\n")
    while True:
        try:
            question = input("Kérdés: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nKilépés.")
            break
        if question.lower() in {"exit", "quit", "kilépés"}:
            break
        if not question:
            continue
        try:
            result = answer_question(question)
        except Exception as exc:
            print(f"\nHiba: {exc}\n")
            continue
        print(f"\nVálasz státusza: {result['status']}")
        print(result["answer"])
        if result["sources"]:
            print("\nForrások:")
            for source in result["sources"]:
                print(
                    f"- [{source['citation_id']}] {source.get('document_title', '?')} "
                    f"| oldal: {source.get('page', '?')} "
                    f"| szakasz: {source.get('section_id', '?')}"
                )
            if result.get("evidence"):
                print("\nEllenőrzött források:")
                for e in result["evidence"]:
                    print(f"- [{e['source_id']}] {e['quote']}")
        else:
            print("\nNincs ellenőrzött forráshivatkozás.")
        if result.get("diagnostics"):
            print("\nDiagnosztika:")
            for item in result["diagnostics"]:
                print("-", item)
        print("\n" + "-" * 60 + "\n")


if __name__ == "__main__":
    main()
