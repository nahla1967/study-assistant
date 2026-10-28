from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings
from langchain_community.vectorstores import FAISS

from config import (
    DATA_DIR, INDEX_DIR, OLLAMA_BASE_URL, EMBEDDING_MODEL,
    CHUNK_SIZE, CHUNK_OVERLAP,
)


def load_documents():
    """Read every PDF / TXT / MD file in data/ into LangChain Documents."""
    docs = []
    for path in sorted(DATA_DIR.glob("*")):
        if path.suffix.lower() == ".pdf":
            docs.extend(PyPDFLoader(str(path)).load())
        elif path.suffix.lower() in {".txt", ".md"}:
            docs.extend(TextLoader(str(path), encoding="utf-8").load())
    return docs


def split_documents(docs):
    """Cut documents into overlapping chunks."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
    )
    return splitter.split_documents(docs)


def build_index(chunks):
    """Embed every chunk and store the vectors in a FAISS index on disk."""
    embeddings = OllamaEmbeddings(model=EMBEDDING_MODEL, base_url=OLLAMA_BASE_URL)
    index = FAISS.from_documents(chunks, embeddings)
    index.save_local(str(INDEX_DIR))


def main():
    docs = load_documents()
    if not docs:
        print(f"No documents found in {DATA_DIR}. Add a PDF/TXT/MD file first.")
        return
    chunks = split_documents(docs)
    print(f"{len(docs)} pages/documents -> {len(chunks)} chunks")
    build_index(chunks)
    print(f"Index saved in {INDEX_DIR}")


if __name__ == "__main__":
    main()