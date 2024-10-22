import os
import numpy as np
import pandas as pd
import faiss
from typing import List

class VectorStore:
    def __init__(self, index_path: str = "faiss_index.bin"):
        self.index_path = index_path
        self.index = None
        self.pages_and_chunks = []

    def load_or_create_index(self, dimension: int):
        if os.path.exists(self.index_path):
            print("Loading existing FAISS index...")
            self.index = faiss.read_index(self.index_path)
        else:
            print("Creating new FAISS index...")
            self.index = faiss.IndexFlatIP(dimension)

    def add_embeddings_from_csv(self, csv_path: str):
        df = pd.read_csv(csv_path)
        df["embedding"] = df["embedding"].apply(lambda x: np.fromstring(x.strip("[]"), sep=" "))
        
        embeddings = np.array(df["embedding"].tolist()).astype('float32')
        self.index.add(embeddings)
        
        self.pages_and_chunks.extend(df.to_dict(orient="records"))
        
        print(f"Added {len(embeddings)} vectors from {csv_path}")

    def save_index(self):
        faiss.write_index(self.index, self.index_path)
        print(f"Saved index to {self.index_path}")

    def search(self, query_vector: np.ndarray, k: int) -> List[dict]:
        scores, indices = self.index.search(query_vector.reshape(1, -1), k)
        return [{"chunk": self.pages_and_chunks[i], "score": float(scores[0][j])} for j, i in enumerate(indices[0])]

def create_vector_store(csv_paths: List[str], index_path: str = "faiss_index.bin") -> VectorStore:
    vector_store = VectorStore(index_path)
    
    first_df = pd.read_csv(csv_paths[0])
    first_embedding = np.fromstring(first_df["embedding"].iloc[0].strip("[]"), sep=" ")
    dimension = len(first_embedding)
    
    vector_store.load_or_create_index(dimension)
    
    for csv_path in csv_paths:
        vector_store.add_embeddings_from_csv(csv_path)
    
    vector_store.save_index()
    return vector_store