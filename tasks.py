import json
from pathlib import Path

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama import ChatOllama

from config import DATA_DIR, LLM_MODEL, OLLAMA_BASE_URL, TEMPERATURE
from ingest import load_documents, split_documents

BATCH = 4          # chunks per summarization call (~4000 chars)
MAX_CHUNKS = 24    # cap on chunks used, to keep local generation fast

llm = ChatOllama(model=LLM_MODEL, base_url=OLLAMA_BASE_URL, temperature=TEMPERATURE)
quiz_llm = ChatOllama(model=LLM_MODEL, base_url=OLLAMA_BASE_URL,
                      temperature=TEMPERATURE, format="json")

MAP_PROMPT = ChatPromptTemplate.from_template(
    "Summarize the excerpts below in 3-4 sentences. Use only the text given, "
    "and write in the same language as the text.\n\n{text}"
)
REDUCE_PROMPT = ChatPromptTemplate.from_template(
    "Below are partial summaries of one document. Merge them into a single "
    "summary: one short intro sentence, then 5-7 bullet points. Use only the "
    "information given, in the same language.\n\n{text}"
)
QUIZ_PROMPT = ChatPromptTemplate.from_template(
    "Write {n} multiple-choice questions based ONLY on the text below.\n"
    'Return JSON in exactly this shape: {{"questions": [{{"question": "...", '
    '"options": ["...", "...", "...", "..."], "answer": "A", "explanation": "..."}}]}}\n'
    "Rules: 4 options per question, exactly one correct, answer is a letter "
    "A-D, same language as the text.\n\nText:\n{context}"
)

QUIZ_WORDS = ("quiz", "quizz", "qcm", "اختبار")
SUMMARY_WORDS = ("summary", "summarize", "summarise", "résumé", "résume", "ملخص", "لخص")


def detect_intent(text):
    """Keyword routing: returns 'quiz', 'summary' or 'question'."""
    t = text.lower()
    if any(w in t for w in QUIZ_WORDS):
        return "quiz"
    if any(w in t for w in SUMMARY_WORDS):
        return "summary"
    return "question"


def list_sources():
    return sorted(p.name for p in DATA_DIR.glob("*")
                  if p.suffix.lower() in {".pdf", ".txt", ".md"})


def get_chunks(source=None):
    chunks = split_documents(_drop_back_matter(load_documents()))
    if source:
        chunks = [c for c in chunks
                  if Path(c.metadata.get("source", "")).name == source]
    return chunks
def _drop_back_matter(docs):
    """Per file, drop the page with a 'References' heading and everything after it."""
    kept, done = [], set()
    for d in docs:
        src = d.metadata.get("source", "")
        if src in done:
            continue
        lines = [l.strip().lower() for l in d.page_content.splitlines()]
        if "references" in lines:
            done.add(src)
            continue
        kept.append(d)
    return kept

def _spread(chunks, n):
    """Pick n chunks evenly spaced across the document, in order."""
    if len(chunks) <= n:
        return chunks
    step = len(chunks) / n
    return [chunks[int(i * step)] for i in range(n)]


def make_summary(source=None):
    chunks = _spread(get_chunks(source), MAX_CHUNKS)
    if not chunks:
        return "No documents found."
    map_chain = MAP_PROMPT | llm | StrOutputParser()
    partials = [
        map_chain.invoke({"text": "\n\n".join(c.page_content for c in chunks[i:i + BATCH])})
        for i in range(0, len(chunks), BATCH)
    ]
    return (REDUCE_PROMPT | llm | StrOutputParser()).invoke(
        {"text": "\n\n".join(partials)})


def _clean(q):
    """Return a validated question dict, or None if the model broke the format."""
    try:
        options = q["options"]
        answer = str(q["answer"]).strip().upper()[:1]
        if len(options) != 4 or answer not in "ABCD" or not q["question"]:
            return None
        return {"question": q["question"], "options": options,
                "answer": answer, "explanation": q.get("explanation", "")}
    except (KeyError, TypeError):
        return None


def make_quiz(source=None, n=5):
    chunks = _spread(get_chunks(source), n)
    if not chunks:
        return []
    context = "\n\n---\n\n".join(c.page_content for c in chunks)
    chain = QUIZ_PROMPT | quiz_llm | StrOutputParser()
    for _ in range(3):  # small models break JSON sometimes: retry
        try:
            data = json.loads(chain.invoke({"n": n, "context": context}))
            quiz = [q for q in map(_clean, data["questions"]) if q]
            if quiz:
                return quiz
        except (json.JSONDecodeError, KeyError, TypeError):
            continue
    return []