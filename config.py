from pathlib import Path

# Paths: built from this file's location, so the project works from any folder
BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data"
INDEX_DIR = BASE_DIR / "faiss_index"

# Ollama models
OLLAMA_BASE_URL = "http://localhost:11434"
LLM_MODEL = "qwen2.5:7b"
EMBEDDING_MODEL = "bge-m3"
TEMPERATURE = 0.2

# Chunking and retrieval
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200
TOP_K = 4