import os
import re
import tqdm
import PyPDF2
import pandas as pd
from sentence_transformers import SentenceTransformer
import chromadb
from chromadb.utils import embedding_functions

def text_formatter(text: str) -> str:
    cleaned_text = text.replace("\n", " ").strip()
    cleaned_text = re.sub(r'\s+', ' ', cleaned_text)
    return cleaned_text

def split_into_sentences(text: str) -> list[str]:
    sentences = re.split(r'(?<=[.!?])\s+', text)
    return [s.strip() for s in sentences if s.strip()]

def open_and_read_pdf(pdf_path: str) -> list[dict]:
    pages_and_texts = []
    with open(pdf_path, 'rb') as file:
        pdf_reader = PyPDF2.PdfReader(file)
        for page_number in tqdm.tqdm(range(len(pdf_reader.pages)), desc="Reading PDF"):
            text = pdf_reader.pages[page_number].extract_text()
            text = text_formatter(text)
            pages_and_texts.append({
                "page_number": page_number,
                "text": text
            })
    return pages_and_texts

def split_list(input_list: list, slice_size: int, overlap: int = 5) -> list[list[str]]:
    chunks = []
    for i in range(0, len(input_list), slice_size - overlap):
        chunk = input_list[i:i + slice_size]
        if chunk:
            chunks.append(chunk)
    return chunks

def chunk_pdf_text(pages_and_texts: list[dict], num_sentence_chunk_size: int = 10, overlap: int = 3) -> pd.DataFrame:
    for item in pages_and_texts:
        item["sentences"] = split_into_sentences(item["text"])

    pages_and_chunks = []
    for item in pages_and_texts:
        item["sentence_chunks"] = split_list(
            input_list=item["sentences"],
            slice_size=num_sentence_chunk_size,
            overlap=overlap
        )
        for sentence_chunk in item["sentence_chunks"]:
            joined_sentence_chunk = " ".join(sentence_chunk)
            joined_sentence_chunk = re.sub(r'\s+', ' ', joined_sentence_chunk)
            chunk_dict = {
                "page_number": item["page_number"],
                "sentence_chunk": joined_sentence_chunk,
                "chunk_char_count": len(joined_sentence_chunk),
                "chunk_word_count": len(joined_sentence_chunk.split(" ")),
                "chunk_token_count": len(joined_sentence_chunk) / 4.0
            }
            pages_and_chunks.append(chunk_dict)

    df = pd.DataFrame(pages_and_chunks)
    return df

def embed_chunks(df: pd.DataFrame, embedding_model_name: str = "all-mpnet-base-v2", min_token_length: int = 30) -> list[dict]:
    filtered = df[df["chunk_token_count"] > min_token_length].to_dict(orient="records")
    embedding_model = SentenceTransformer(embedding_model_name, device="cpu")

    for item in tqdm.tqdm(filtered, desc="Generating Embeddings"):
        item["embedding"] = embedding_model.encode(item["sentence_chunk"])

    return filtered

def store_in_chroma(chunks: list[dict], collection_name: str, chroma_db_path: str = "./data/chroma_db", embedding_model_name: str = "all-mpnet-base-v2") -> None:
    chroma_client = chromadb.PersistentClient(path=chroma_db_path)

    embedding_function = embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name=embedding_model_name
    )

    collection = chroma_client.get_or_create_collection(
        name=collection_name,
        embedding_function=embedding_function,
        metadata={"hnsw:space": "cosine"}
    )

    ids = [str(i) for i in range(len(chunks))]
    documents = [item["sentence_chunk"] for item in chunks]
    metadatas = [{"page_number": str(item["page_number"])} for item in chunks]
    embeddings = [item["embedding"].tolist() for item in chunks]

    collection.upsert(
        ids=ids,
        documents=documents,
        metadatas=metadatas,
        embeddings=embeddings
    )
    print(f"Embeddings stored in Chroma collection '{collection_name}' at {chroma_db_path}.")

def embed_pdf_into_chroma(pdf_path: str, collection_name: str = "pdf_embeddings", chroma_db_path: str = "./data/chroma_db",
                          embedding_model_name: str = "all-mpnet-base-v2") -> None:
    if not os.path.exists(pdf_path):
        raise FileNotFoundError(f"File '{pdf_path}' doesn't exist.")

    pages_and_texts = open_and_read_pdf(pdf_path=pdf_path)
    df_chunks = chunk_pdf_text(pages_and_texts)
    chunks = embed_chunks(df_chunks, embedding_model_name=embedding_model_name)
    store_in_chroma(chunks, collection_name=collection_name, chroma_db_path=chroma_db_path, embedding_model_name=embedding_model_name)
    print("PDF embedding process complete.")

if __name__ == "__main__":
    pdf_path = "data/pdfs/example.pdf"  # Update this to your actual PDF
    collection_name = "pdf_embeddings"
    embedding_model_name = "all-mpnet-base-v2"
    chroma_db_path = "./data/chroma_db"

    embed_pdf_into_chroma(pdf_path=pdf_path, collection_name=collection_name, chroma_db_path=chroma_db_path, embedding_model_name=embedding_model_name)
