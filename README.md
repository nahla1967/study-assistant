# AI Study Assistant

A local RAG app that answers questions about your own course notes and papers. You upload PDFs, text or Markdown files, and it answers using only what is in them, with the source pages shown under each answer.

Everything runs on your machine through Ollama, so your documents never leave it.

## How it works

1. Uploaded files are loaded and split into overlapping chunks.
2. Each chunk is embedded with `<embedding-model>` and stored in a local FAISS index.
3. At question time, the question is embedded and the closest chunks are retrieved.
4. Those chunks are put into the prompt, and `<chat-model>` answers from them only. If the context doesn't contain the answer, it says so instead of guessing.

## Stack

Python 3.11, LangChain, Ollama (chat and embeddings), FAISS, Streamlit, pypdf.

## Setup

You need [Ollama](https://ollama.com) installed and running.

```powershell
git clone https://github.com/nahla1967/study-assistant.git
cd study-assistant
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt

ollama pull <chat-model>
ollama pull <embedding-model>

streamlit run app.py
```

Upload your files in the sidebar, click **Process documents**, then ask questions in the chat.

## Project structure

```
app.py      Streamlit interface (upload, indexing, chat, sources)
config.py   Paths, model names, chunk size, top-k
ingest.py   Loading, splitting and building the FAISS index
rag.py      Retrieval and answer generation
```

## Limitations

- Questions about document metadata ("who wrote this paper?") often fail. The retrieved chunks tend to come from the reference list, which is full of author names, while the title page never ranks in the top results. Content questions work much better.
- Indexing re-embeds every file in `data/` each time, so it slows down as the collection grows.
- Answer quality depends on the local model. Small models follow instructions less reliably than hosted ones.

## Roadmap

- Summaries and quizzes generated from the same retrieval
- Incremental indexing
