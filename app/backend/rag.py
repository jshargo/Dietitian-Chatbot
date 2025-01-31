import os
import pandas as pd
import numpy as np
import logging
from sentence_transformers import SentenceTransformer

logger = logging.getLogger(__name__)

HUGGINGFACE_API_KEY = os.getenv("HUGGINGFACE_API_KEY")
EMBED_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
embedder = SentenceTransformer(EMBED_MODEL_NAME)

# Use Docker path to match potts.py
CSV_PATH = "/app/data/intent_embeddings/intent_embeddings_all.csv"

df_knowledge = pd.DataFrame()
try:
    df_knowledge = pd.read_csv(CSV_PATH)
    logger.info("Knowledge CSV loaded successfully.")
except Exception as e:
    logger.warning(f"Could not load knowledge CSV: {e}")

def retrieve_knowledge(query: str, top_k=3):
    """
    Retrieve top-k relevant chunks from the CSV by embedding + similarity.
    """
    if df_knowledge.empty:
        return ["No knowledge base loaded."]

    query_emb = embedder.encode([query])[0]
    knowledge_vectors = df_knowledge[[c for c in df_knowledge.columns if c.startswith("f")]].values

    # Compute cosine similarity using same method as IntentClassifier
    dot_product = np.dot(knowledge_vectors, query_emb)
    knowledge_norms = np.linalg.norm(knowledge_vectors, axis=1)
    query_norm = np.linalg.norm(query_emb)
    similarity_scores = dot_product / (knowledge_norms * query_norm)

    df_knowledge["similarity"] = similarity_scores

    df_sorted = df_knowledge.sort_values(by="similarity", ascending=False)
    top_rows = df_sorted.head(top_k)

    return top_rows["text"].tolist()