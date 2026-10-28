from langchain_community.vectorstores import FAISS
from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

from config import (
    INDEX_DIR, OLLAMA_BASE_URL, EMBEDDING_MODEL,
    LLM_MODEL, TEMPERATURE, TOP_K,
)

PROMPT = ChatPromptTemplate.from_messages([
    ("system",
     "You are a study assistant. Answer the question using ONLY the context "
     "provided. If the context does not contain the answer, say that the "
     "documents do not cover it. Do not use outside knowledge. "
     "Reply in the same language as the question."),
    ("human", "Context:\n{context}\n\nQuestion: {question}"),
])


def load_index():
    """Load the FAISS index saved by ingest.py."""
    embeddings = OllamaEmbeddings(model=EMBEDDING_MODEL, base_url=OLLAMA_BASE_URL)
    return FAISS.load_local(
        str(INDEX_DIR), embeddings, allow_dangerous_deserialization=True
    )


def get_llm():
    return ChatOllama(
        model=LLM_MODEL, base_url=OLLAMA_BASE_URL, temperature=TEMPERATURE
    )


def format_context(chunks):
    """Join chunks into one text block, labelled with their source."""
    parts = []
    for chunk in chunks:
        source = chunk.metadata.get("source", "unknown")
        page = chunk.metadata.get("page")
        label = f"{source} (page {page + 1})" if page is not None else source
        parts.append(f"[{label}]\n{chunk.page_content}")
    return "\n\n---\n\n".join(parts)


def answer_question(question, index=None):
    """Retrieve the closest chunks, then ask the LLM to answer from them."""
    index = index or load_index()
    chunks = index.max_marginal_relevance_search(question, k=TOP_K, fetch_k=20)
    chain = PROMPT | get_llm() | StrOutputParser()
    answer = chain.invoke(
        {"context": format_context(chunks), "question": question}
    )
    return answer, chunks


if __name__ == "__main__":
    q = input("Question: ")
    ans, sources = answer_question(q)
    print("\nAnswer:\n", ans)
    print("\nRetrieved chunks:")
    for s in sources:
        print("-" * 60)
        print("page", s.metadata.get("page", 0) + 1)
        print(s.page_content[:400])