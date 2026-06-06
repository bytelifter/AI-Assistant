from __future__ import annotations

import streamlit as st

from core.config import AppConfig
from core.exceptions import BotError, KnowledgeBaseNotReadyError
from services import KnowledgeService


st.set_page_config(page_title="Engine SpA - AI Assistant", page_icon="🚦", layout="wide")


@st.cache_resource(show_spinner=False)
def get_service() -> KnowledgeService:
    config = AppConfig.from_env()
    return KnowledgeService(config)


def _render_sources(sources: list[str]) -> None:
    if not sources:
        return
    with st.expander("Fonti utilizzate", expanded=False):
        for source in sources:
            st.write(f"- {source}")


def main() -> None:
    st.title("💬 Area Pre-Sales - Engine SpA")
    st.caption("Motore RAG con database vettoriale persistente per ricerca rapida nei documenti interni.")

    try:
        service = get_service()
    except KnowledgeBaseNotReadyError as exc:
        st.error(str(exc))
        st.info("Esegui prima `python ingest.py` per creare o aggiornare l'indice vettoriale.")
        st.stop()
    except Exception as exc:
        st.error(f"Configurazione non valida: {exc}")
        st.stop()

    if "messages" not in st.session_state:
        st.session_state.messages = []

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    prompt = st.chat_input("Chiedi info tecniche sui prodotti...")
    if not prompt:
        return

    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Cerco nelle fonti e preparo la risposta..."):
            try:
                result = service.ask(prompt, history=st.session_state.messages[:-1])
                st.markdown(result.answer)
                _render_sources(result.sources)
                st.session_state.messages.append({"role": "assistant", "content": result.answer})
            except BotError as exc:
                st.error(str(exc))
            except Exception as exc:
                st.error(f"Errore imprevisto: {exc}")


if __name__ == "__main__":
    main()
