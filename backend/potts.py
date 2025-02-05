import pandas as pd
import numpy as np
from typing import Tuple, List

""" 12/15/2024: nt
Module for Ms. Potts.  So far it classifies a user query into one of the four intents:
	0. Meal-Logging
	1. Meal-Planning-Recipes
	2. Educational-Content
	3. Personalized-Health-Advice
"""

CSV_PATH = "../data/intent_embeddings/intent_embeddings_all.csv"

class IntentClassifier:
    def __init__(self):
        self.intent_df = pd.read_csv(CSV_PATH)
        
    def compute_similarity(self, query_embedding: np.ndarray, intent_embeddings: np.ndarray) -> List[Tuple[int, float]]:
        similarities = np.dot(intent_embeddings, query_embedding) / (
            np.linalg.norm(intent_embeddings, axis=1) * np.linalg.norm(query_embedding)
        )
        
        top_indices = np.argsort(similarities)[::-1]
        return [(idx, similarities[idx]) for idx in top_indices[:3]]
    
    def classify_from_embedding(self, query_embedding: np.ndarray) -> dict:
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