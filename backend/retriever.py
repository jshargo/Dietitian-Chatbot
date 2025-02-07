# retriever.py
import numpy as np
import pandas as pd
import os
import logging
from sentence_transformers import SentenceTransformer

logger = logging.getLogger(__name__)

class Retriever:
    def __init__(self):
        """Initialize the retriever by loading the embedding model and the knowledge base."""
        try:
            self.embed_model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
            
            # Construct the path to the CSV file. Adjust the relative path if needed.
            data_path = os.path.join(os.path.dirname(__file__), '../data/embeddings.csv')
            self.knowledge_df = pd.read_csv(data_path)
            
            # Convert string representations of embeddings into numpy arrays.
            self.knowledge_embeddings = np.array(
                self.knowledge_df['embedding']
                .apply(lambda x: np.array(eval(x), dtype=np.float32))
                .tolist()
            )
            self.knowledge_texts = self.knowledge_df['sentence_chunk'].tolist()
        except Exception as e:
            logger.error(f"Error initializing Retriever: {e}")
            raise

    def embed_query(self, query: str) -> np.ndarray:
        """Generate an embedding for the given query."""
        return self.embed_model.encode([query])[0]

    def retrieve(self, query: str) -> str:
        """Retrieve relevant context from the knowledge base based on the query."""
        try:
            query_embedding = self.embed_query(query)
            query_norm = query_embedding / np.linalg.norm(query_embedding)
            knowledge_norms = np.linalg.norm(self.knowledge_embeddings, axis=1)
            similarities = np.dot(self.knowledge_embeddings, query_norm) / knowledge_norms
            max_score = np.max(similarities)
            
            if max_score < 0.30:
                return (f"OUT_OF_SCOPE: This question is outside my nutrition expertise. "
                        f"Please ask about food, nutrients, or health-related topics. (score: {max_score:.2f})")
                
            most_relevant_idx = np.argmax(similarities)
            return f"Knowledge Source (score: {max_score:.2f}): {self.knowledge_texts[most_relevant_idx]}"
            
        except Exception as e:
            logger.error(f"Retrieval error: {e}")
            return "Error accessing knowledge base"