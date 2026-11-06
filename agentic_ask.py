"""Agentic implementation of CLI-based chat of OTP Diákhitel számla"""
from src.graph import ask_agentic


def main() -> None:
    print("OTP Diákhitel Számla - agent (kilépés: exit)\n")
    while True:
        try:
            question = input("Kérdés: ").strip()
        except (KeyboardInterrupt, EOFError):
            break
        if question.casefold() in {"exit", "quit", "kilépés"}:
            break
        if not question:
            continue
        try:
            result = ask_agentic(question)
        except Exception as exc:
            print("Hiba:", exc)
            continue
        print("\nStátusz:", result.get("status"))
        print("Mód:", result.get("mode", "-"))
        print(result.get("answer", ""))
        for item in result.get("sources", []):
            print(
                f"  [{item.get('citation_id', '-')}] "
                f"{item.get('document_title', '?')} | "
                f"oldal: {item.get('page', '?')} | "
                f"szakasz: {item.get('section_id', '?')}"
            )
        print("Gráf lépései:", " -> ".join(result.get("graph_trace", [])))
        print("Újraírások száma:", result.get("rewrite_count", 0))
        print("-" * 60, "\n")


if __name__ == "__main__":
    main()
