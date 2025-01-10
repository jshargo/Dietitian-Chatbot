import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
from typing import List, Tuple
import os

""" 12/15/2024: nt
Module for Ms. Potts.  So far it classifies a user query into one of the four intents:
	0. Meal-Logging
	1. Meal-Planning-Recipes
	2. Educational-Content
	3. Personalized-Health-Advice
"""

class IntentClassifier:
    def __init__(self):
        self.model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
        
        base_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data')
        self.intent_df = pd.read_csv(os.path.join(base_path, 'intent_embeddings/intent_embeddings_all.csv'))
        
    def embed_query(self, query: str) -> np.ndarray:
        return self.model.encode([query])[0]
    
    def compute_similarity(self, query_embedding: np.ndarray, intent_embeddings: np.ndarray) -> List[Tuple[int, float]]:
        similarities = np.dot(intent_embeddings, query_embedding) / (
            np.linalg.norm(intent_embeddings, axis=1) * np.linalg.norm(query_embedding)
        )
        
        top_indices = np.argsort(similarities)[::-1]
        return [(idx, similarities[idx]) for idx in top_indices[:3]]
    
    def classify(self, query: str) -> dict:
        query_embedding = self.embed_query(query)
        
        df_temp = self.intent_df.drop(['Intent', 'Category'], axis=1)
        embeddings = df_temp.to_numpy()
        
        results = self.compute_similarity(query_embedding, embeddings)
        
        categories = self.intent_df['Category'].to_numpy()
        intents = self.intent_df['Intent'].to_numpy()
        
        classifications = [
            {
                "category": categories[idx],
                "intent": intents[idx],
                "confidence": float(score)
            }
            for idx, score in results
        ]
        
        return {
            "top_intent": classifications[0]["intent"],
            "top_category": classifications[0]["category"],
            "classifications": classifications
        }