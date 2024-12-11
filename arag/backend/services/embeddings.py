from sentence_transformers import SentenceTransformer

def load_embedding_model(path: str):
    model = SentenceTransformer(path)
    return model

def get_embedding(model, text: str):
    return model.encode([text])[0]
