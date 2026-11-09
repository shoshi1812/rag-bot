"""All project settings in one place. Each value maps to a decision from the syllabus."""
import os

import windows_certificates  # noqa: F401  (first! fixes HTTPS behind the NetFree filter)
from dotenv import load_dotenv

# override=True: values in .env win over variables already set in Windows
load_dotenv(override=True)

# Decision 5 + 7: API key and models (two different models, two different jobs)
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
EMBED_MODEL = os.getenv("EMBED_MODEL", "gemini-embedding-001")  # text -> vector
LLM_MODEL = os.getenv("LLM_MODEL", "gemini-3.1-flash-lite")      # writes the answer

# Decision 1: Vector DB (Chroma, saved on disk)
DATA_DIR = "data"
DB_DIR = "storage"
COLLECTION = "procedures"

# Decision 3 + 4: chunking by section, with a size cap and a small overlap
CHUNK_SIZE = 400     # tokens
CHUNK_OVERLAP = 50   # tokens

# Decision 6: how many chunks are sent to the LLM
TOP_K = 4

# Decision 8: expand only short questions
SHORT_QUESTION_WORDS = 4

# Low temperature = answers that stick to the documents
TEMPERATURE = 0.1
