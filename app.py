from pathlib import Path

import streamlit as st

from config import DATA_DIR
from ingest import load_documents, split_documents, build_index
from rag import load_index, answer_question

st.set_page_config(page_title="AI Study Assistant", page_icon="📚")
st.title("📚 AI Study Assistant")


@st.cache_resource
def get_index():
    """Load the FAISS index once and reuse it across reruns."""
    return load_index()


# ---------- Sidebar: upload and index ----------
with st.sidebar:
    st.header("Your documents")
    uploaded = st.file_uploader(
        "Upload PDF, TXT or MD", type=["pdf", "txt", "md"],
        accept_multiple_files=True,
    )
    if uploaded and st.button("Process documents"):
        DATA_DIR.mkdir(exist_ok=True)
        for f in uploaded:
            (DATA_DIR / f.name).write_bytes(f.getbuffer())
        with st.spinner("Indexing..."):
            chunks = split_documents(load_documents())
            build_index(chunks)
            get_index.clear()  # forget the old index so the new one is loaded
        st.success(f"Indexed {len(chunks)} chunks")

# ---------- Chat ----------
if "messages" not in st.session_state:
    st.session_state.messages = []

for m in st.session_state.messages:
    with st.chat_message(m["role"]):
        st.markdown(m["content"])

question = st.chat_input("Ask a question about your documents")
if question:
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        try:
            index = get_index()
        except Exception:
            st.error("No index found. Upload and process a document first.")
            st.stop()

        with st.spinner("Thinking..."):
            answer, sources = answer_question(question, index)
        st.markdown(answer)

        with st.expander("Sources"):
            for s in sources:
                name = Path(s.metadata.get("source", "unknown")).name
                page = s.metadata.get("page")
                label = f"{name}, page {page + 1}" if page is not None else name
                st.caption(label)
                st.text(s.page_content[:300])

    st.session_state.messages.append({"role": "assistant", "content": answer})