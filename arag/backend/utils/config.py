import os

DATABASE_URL = os.getenv("DATABASE_URL", "postgresql://user:password@postgres:5432/dbname")
SECRET_KEY = os.getenv("SECRET_KEY", "supersecret")
CHROMA_PATH = os.getenv("CHROMA_PATH", "/app/chroma")
EMBEDDING_MODEL_PATH = os.getenv("EMBEDDING_MODEL_PATH", "/app/models/embedding-model")
LLM_MODEL_PATH = os.getenv("LLM_MODEL_PATH", "/app/models/llm-model")
