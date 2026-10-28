import re
from pathlib import Path

import streamlit as st

from config import DATA_DIR
from ingest import load_documents, split_documents, build_index
from rag import load_index, answer_question
from tasks import list_sources, make_summary, make_quiz, detect_intent

st.set_page_config(page_title="AI Study Assistant", page_icon="📚")
st.title("📚 AI Study Assistant")


@st.cache_resource
def get_index():
    """Load the FAISS index once and reuse it across reruns."""
    return load_index()


# ---------- Sidebar: upload, index, choose document ----------
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

    sources = list_sources()
    choice = st.selectbox("Document", ["All documents"] + sources) if sources else None
    source = None if choice in (None, "All documents") else choice

# ---------- Main area ----------
if not list_sources():
    st.info("Upload a document in the sidebar, then click Process documents.")
    st.stop()

for key, default in [("messages", []), ("summary", ""), ("quiz", []), ("quiz_v", 0)]:
    if key not in st.session_state:
        st.session_state[key] = default

tab_chat, tab_sum, tab_quiz = st.tabs(["💬 Ask", "📝 Summary", "🧠 Quiz"])


def strip_letter(opt):
    """Remove a leading 'A) ' or 'B. ' the model may have put in the option."""
    return re.sub(r"^[A-D][\).]\s*", "", opt)


# ---------- Summary tab ----------
with tab_sum:
    if st.button("Generate summary"):
        with st.spinner("Summarizing..."):
            st.session_state.summary = make_summary(source)
    if st.session_state.summary:
        st.markdown(st.session_state.summary)

# ---------- Quiz tab ----------
with tab_quiz:
    n = st.slider("Number of questions", 3, 10, 5)
    if st.button("Generate quiz"):
        with st.spinner("Writing questions..."):
            st.session_state.quiz = make_quiz(source, n)
        st.session_state.quiz_v += 1
        if not st.session_state.quiz:
            st.error("The model did not return a valid quiz. Try again.")

    quiz, v = st.session_state.quiz, st.session_state.quiz_v
    for i, q in enumerate(quiz):
        st.markdown(f"**{i + 1}. {q['question']}**")
        st.radio("Answer", [strip_letter(o) for o in q["options"]],
                 index=None, key=f"q{v}_{i}", label_visibility="collapsed")

    if quiz and st.button("Check answers"):
        score = 0
        for i, q in enumerate(quiz):
            correct = strip_letter(q["options"]["ABCD".index(q["answer"])])
            if st.session_state.get(f"q{v}_{i}") == correct:
                score += 1
            else:
                st.warning(f"Q{i + 1}: correct answer is **{correct}**. {q['explanation']}")
        st.success(f"Score: {score}/{len(quiz)}")

# ---------- Ask tab ----------
with tab_chat:
    for m in st.session_state.messages:
        with st.chat_message(m["role"]):
            st.markdown(m["content"])

    question = st.chat_input("Ask a question, or type 'quiz' / 'summary'")
    if question:
        st.session_state.messages.append({"role": "user", "content": question})
        intent = detect_intent(question)

        if intent in ("quiz", "summary"):
            with st.spinner("Working on it..."):
                if intent == "quiz":
                    st.session_state.quiz = make_quiz(source)
                    st.session_state.quiz_v += 1
                    ok, where = bool(st.session_state.quiz), "🧠 Quiz"
                else:
                    st.session_state.summary = make_summary(source)
                    ok, where = bool(st.session_state.summary), "📝 Summary"
            reply = f"Done. Open the {where} tab." if ok else "That failed, please try again."
            st.session_state.messages.append({"role": "assistant", "content": reply})
            st.rerun()

        with st.chat_message("user"):
            st.markdown(question)
        with st.chat_message("assistant"):
            try:
                index = get_index()
            except Exception:
                st.error("No index found. Process the documents first.")
                st.stop()
            with st.spinner("Thinking..."):
                answer, srcs = answer_question(question, index)
            st.markdown(answer)
            with st.expander("Sources"):
                for s in srcs:
                    name = Path(s.metadata.get("source", "unknown")).name
                    page = s.metadata.get("page")
                    st.caption(f"{name}, page {page + 1}" if page is not None else name)
                    st.text(s.page_content[:300])
        st.session_state.messages.append({"role": "assistant", "content": answer})