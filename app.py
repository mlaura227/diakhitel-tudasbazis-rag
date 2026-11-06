"""Local Streamlit chat UI for the established Basic and Agentic RAG pipelines.

Run from the existing project root: python -m streamlit run app.py
The app does not connect to an external hosted LLM service.
"""
from __future__ import annotations

from time import perf_counter

import streamlit as st

from src.ui_helpers import MODE_NAMES, diagnostic_json, history_as_csv, result_summary, source_line


st.set_page_config(page_title="OTP Diákhitel Számla - lokál RAG", page_icon="🏦", layout="wide")


def run_one(question: str, engine: str) -> dict:
    if engine == "basic":
        from src.generation import answer_question
        return answer_question(question)
    if engine == "agentic":
        from src.graph import ask_agentic
        return ask_agentic(question)
    raise ValueError(f"Unknown engine: {engine}")


def render_result(result: dict, engine: str, seconds: float, key: str) -> None:
    st.markdown(f"**{MODE_NAMES[engine]}**")
    status = str(result.get("status", "unknown"))
    if status == "insufficient":
        st.warning(result.get("answer") or "A források alapján nincs megbízható válasz.")
    elif status == "partial":
        st.info(result.get("answer") or "Részleges válasz.")
    else:
        st.markdown(result.get("answer") or "Nincs válasz.")

    st.caption(f"{result_summary(result)} · {seconds:.2f} s")

    facet_trace = result.get("facet_trace") or []
    if facet_trace:
        with st.expander(f"Kérdés ellenőrzése ({len(facet_trace)})", expanded=False):
            for facet in facet_trace:
                st.write(f"**{facet.get('status', '?')}** · {facet.get('facet', '')}")
            missing = result.get("missing_facets") or []
            if missing:
                st.warning(f"{len(missing)} kérdés forrása hiányzik. A részleges választ nem jelöljük teljesnek.")
    if result.get("applicability_issue"):
        st.caption(f"Díjalkalmazhatósági védelem: {result['applicability_issue']}")

    sources = result.get("sources") or []
    evidence = result.get("evidence") or []
    with st.expander(f"Források és idézetek ({len(sources)})", expanded=bool(sources)):
        if sources:
            for source in sources:
                st.markdown(f"**{source_line(source)}**")
        else:
            st.write("Nincs ellenőrzött forráshivatkozás.")
        if evidence:
            st.markdown("**Ellenőrzött szövegrészletek**")
            for item in evidence:
                st.caption(f"[{item.get('source_id', '?')}]")
                st.code(str(item.get("quote", "")), language=None, wrap_lines=True)

    if engine == "agentic":
        steps = result.get("graph_trace") or []
        with st.expander("LangGraph végrehajtás", expanded=False):
            st.code(" → ".join(steps) or "Nincs gráfnyomvonal", language=None)
            st.write("Lekérdezések száma:", result.get("retrieval_rounds", "?"))
            st.write("Újraírások:", result.get("rewrite_count", "?"))
            st.write("Retrieval osztályozás:", result.get("retrieval_grade", "?"))
            st.caption(result.get("retrieval_grade_reason", ""))
            st.caption("A retrieval osztályozás nem igazolja a végső válasz ténybeli helyességét.")

    if st.session_state.show_diagnostics:
        with st.expander("Részletes diagnosztika", expanded=False):
            st.code(diagnostic_json(result), language="json")


def render_turn(turn: dict, index: int) -> None:
    with st.chat_message("user"):
        st.markdown(turn["question"])
    with st.chat_message("assistant"):
        results = turn.get("results", {})
        if len(results) == 2:
            cols = st.columns(2)
            for col, engine in zip(cols, ("basic", "agentic")):
                with col:
                    result = results.get(engine)
                    if result:
                        render_result(result, engine, turn["times"][engine], f"{index}-{engine}")
        else:
            for engine, result in results.items():
                render_result(result, engine, turn["times"][engine], f"{index}-{engine}")


if "history" not in st.session_state:
    st.session_state.history = []
if "show_diagnostics" not in st.session_state:
    st.session_state.show_diagnostics = False

st.title("🏦 OTP Diákhitel Számla - lokál RAG")
st.caption("Ollama (Qwen 2.5 3B + BGE-M3) · ChromaDB · Basic RAG / LangGraph")
st.info(
    "2026.novemberében hatályos lakossági üzletszabályzat, díjjegyzék, hirdetmény, "
    "valamint glosszárium dokumentumok alapján kérdezhetsz az OTP Diákhitel számláról."
)

with st.sidebar:
    st.header("Beállítások")
    selected_mode = st.radio(
        "Futtatási mód",
        options=("basic", "agentic", "compare"),
        format_func=lambda v: MODE_NAMES[v],
        index=1,
    )
    st.checkbox("Technikai diagnosztika", key="show_diagnostics")
    st.caption("Az összehasonlító mód mindkét rendszert külön futtatja, egymás után.")
    if st.button("Beszélgetés törlése"):
        st.session_state.history = []
        st.rerun()
    if st.session_state.history:
        st.download_button(
            "Beszélgetés exportálása (CSV)",
            data=("\ufeff" + history_as_csv(st.session_state.history)).encode("utf-8"),
            file_name="otp_rag_session.csv",
            mime="text/csv",
        )
    st.divider()
    st.caption("Az export logoltan tartalmazza a beszélgetést.")

for i, previous in enumerate(st.session_state.history):
    render_turn(previous, i)

question = st.chat_input("Kérdezz a Diákhitel számla díjairól vagy fogalmairól…")
if question and question.strip():
    clean_question = question.strip()
    with st.chat_message("user"):
        st.markdown(clean_question)
    engines = ("basic", "agentic") if selected_mode == "compare" else (selected_mode,)
    results = {}
    times = {}
    with st.chat_message("assistant"):
        with st.spinner("Forráskeresés és válaszkészítés folyamatban..."):
            try:
                for engine in engines:
                    start = perf_counter()
                    results[engine] = run_one(clean_question, engine)
                    times[engine] = perf_counter() - start
            except Exception as exc:
                st.error(f"Hiba a lokál RAG futtatása közben: {type(exc).__name__}: {exc}")
            else:
                turn = {"question": clean_question, "results": results, "times": times}
                st.session_state.history.append(turn)
                if len(results) == 2:
                    cols = st.columns(2)
                    for col, engine in zip(cols, ("basic", "agentic")):
                        with col:
                            render_result(results[engine], engine, times[engine], f"new-{engine}")
                else:
                    engine = engines[0]
                    render_result(results[engine], engine, times[engine], "new")
