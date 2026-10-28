# AI Study Assistant

A local RAG app for studying from your own documents. Upload PDFs, text or Markdown files, then ask questions, get a summary, or take a generated quiz. Answers come only from your files, and the source pages are shown under each answer.

Everything runs on your machine through Ollama, so your documents never leave it.

## Features

- **Ask:** question answering grounded in the retrieved chunks, with sources. If the context doesn't contain the answer, the model says so instead of guessing.
- **Summary:** a map-reduce summary. Chunks sampled evenly across the document are summarized in small batches, then the partial summaries are merged. Plain retrieval can't do this, because it returns the chunks closest to a query rather than coverage of the whole document.
- **Quiz:** multiple-choice questions generated as JSON, validated, and retried when the local model breaks the format. Answers are checked in the app with explanations.
- **Intent routing:** typing "quiz" or "summary" in the chat triggers the matching feature. The keyword lists include French and Arabic terms.
- **Document selector:** run a summary or quiz on one file or on all of them.

## How it works

1. Files are split into chunks of 1000 characters with 200 overlap.
2. Each chunk is embedded with `bge-m3` and stored in a local FAISS index.
3. For a question, the 4 closest chunks are retrieved and placed in the prompt, and `qwen2.5:7b` answers from them.

## Stack

Python 3.11, LangChain, Ollama, FAISS, Streamlit, pypdf.

## Setup

Requires [Ollama](https://ollama.com) installed and running.

```powershell
git clone https://github.com/nahla1967/study-assistant.git
cd study-assistant
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt

ollama pull qwen2.5:7b
ollama pull bge-m3

streamlit run app.py
```

Upload files in the sidebar, click **Process documents**, then use the three tabs. Models and chunk settings can be changed in `config.py`.

## Project structure

```
app.py      Streamlit interface (upload, tabs, chat)
config.py   Paths, model names, chunking and retrieval settings
ingest.py   Loading, splitting and building the FAISS index
rag.py      Retrieval and answer generation
tasks.py    Summary, quiz generation and intent routing
```

## Limitations

- Questions about metadata ("who wrote this paper?") often fail. The closest chunks tend to come from the reference list, which is full of author names, while the title page never ranks in the top results. Content questions work much better.
- Indexing re-embeds every file in `data/` each time, so it slows down as the collection grows.
- Quiz and summary quality depend on the 7B model. Some generated questions are weak and should be checked against the source.
