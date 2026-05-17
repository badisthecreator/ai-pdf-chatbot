import os
from dotenv import load_dotenv

load_dotenv()

# groq api key comes from the .env file
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
GROQ_MODEL = "llama-3.3-70b-versatile"

# local embedding model so we don't need another api key
EMBEDDING_MODEL = "all-MiniLM-L6-v2"

# folder where chromadb saves the vectors
CHROMA_PERSIST_DIR = "./vectorstore"

# each chunk is 1000 chars, and chunks overlap by 200 so we don't lose context at the edges
CHUNK_SIZE    = 1000
CHUNK_OVERLAP = 200

# how many chunks to fetch per question
TOP_K = 5
