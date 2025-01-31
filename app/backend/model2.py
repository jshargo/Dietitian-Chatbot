import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from openai import OpenAI
import os
from dotenv import load_dotenv
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

load_dotenv()

os.environ["TOKENIZERS_PARALLELISM"] = "false"

DEFAULT_MODEL = "gpt-4o-mini"
AGENT_LOOP_LIMIT = 3

class Model:
    def __init__(self):
        try:
            self.client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))
            self.model = os.getenv('OPENAI_MODEL', DEFAULT_MODEL)
            
            self.embedder = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

            self.embeddings_df = pd.read_csv("../data/embeddings.csv")
            
            self.embeddings_df['embedding'] = self.embeddings_df['embedding'].apply(
                lambda x: np.array(eval(x)) if isinstance(x, str) else x
            )
            
            sample_embedding = self.embeddings_df['embedding'].iloc[0]
            model_dim = self.embedder.get_sentence_embedding_dimension()
            
            if len(sample_embedding) != model_dim:
                raise ValueError(f"Embedding dimension mismatch. Model expects {model_dim}D, CSV has {len(sample_embedding)}D embeddings")
            
            self.embeddings = np.stack(self.embeddings_df['embedding'].values)
            self.norms = np.linalg.norm(self.embeddings, axis=1)
            
            self.messages = [
                {"role": "system", "content": """You are a helpful AI assistant specializing in nutrition and diet advice. 
                Format your responses with these rules:
                - Use proper paragraph breaks between main points
                - Use clean markdown formatting without asterisks or any other symbols
                - Use bullet points or numbered lists when appropriate
                - Keep responses concise and well-structured"""}
            ]
            
        except Exception as e:
            logger.error(f"Initialization failed: {e}")
            raise

    def _get_top_contexts(self, query_embedding: np.ndarray, top_k: int = 3) -> list:
        """Get top matching contexts using cosine similarity"""
        query_embedding = query_embedding.flatten().astype(np.float32)
        
        similarities = np.dot(self.embeddings, query_embedding) / (self.norms * np.linalg.norm(query_embedding))
        
        top_indices = np.argsort(similarities)[-top_k:][::-1]
        
        return [{
            "text": self.embeddings_df.iloc[idx]['sentence_chunk'],
            "similarity": float(similarities[idx])
        } for idx in top_indices]

    def get_response(self, query: str) -> dict:
        try:
            query_embedding = self.embedder.encode(query, convert_to_tensor=False, normalize_embeddings=True)
            contexts = self._get_top_contexts(query_embedding)
            context_str = "\n".join([f"Context (similarity: {c['similarity']:.2f}): {c['text']}" for c in contexts])
            
            self.messages.append({"role": "system", "content": f"Consider these relevant contexts:\n{context_str}"})
            self.messages.append({"role": "user", "content": query})

            response = self.client.chat.completions.create(
                model=self.model,
                messages=self.messages,
                temperature=0.3
            )
            
            # Add assistant's response to history
            self.messages.append({"role": "assistant", "content": response.choices[0].message.content})
            
            return {
                "answer": response.choices[0].message.content,
                "contexts": contexts
            }
            
        except Exception as e:
            logger.error(f"Query failed: {e}")
            return {"error": str(e)}
