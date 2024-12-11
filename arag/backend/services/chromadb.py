import chromadb
from chromadb.config import Settings

def get_chroma_client(path):
    client = chromadb.Client(Settings(
        chroma_db_impl="duckdb+parquet",
        persist_directory=path
    ))
    return client

def get_or_create_collection(client, name="pdf_chunks"):
    return client.get_or_create_collection(name)
