import os
import re
import tqdm
import PyPDF2
import pandas as pd
from typing import List, Dict
from sentence_transformers import SentenceTransformer

def text_formatter(text: str) -> str:
    cleaned_text = text.replace("\n", " ").strip()
    cleaned_text = re.sub(r'\s+', ' ', cleaned_text)
    return cleaned_text

def split_into_sentences(text: str) -> List[str]:
    sentences = re.split(r'(?<=[.!?])\s+', text)
    return [s.strip() for s in sentences if s.strip()]

def open_and_read_pdf(pdf_path: str) -> List[Dict]:
    if not os.path.exists(pdf_path):
        raise FileNotFoundError(f"PDF file not found: {pdf_path}")

    with open(pdf_path, 'rb') as file:
        pdf_reader = PyPDF2.PdfReader(file)
        pages_and_texts = []
        for page_number in tqdm.tqdm(range(len(pdf_reader.pages)), desc="Reading PDF"):
            page_text = pdf_reader.pages[page_number].extract_text() or ""
            page_text = text_formatter(page_text)
            pages_and_texts.append({"page_number": page_number, "text": page_text})
    return pages_and_texts

def split_list(input_list: List[str], slice_size: int, overlap: int = 5) -> List[List[str]]:
    chunks = []
    step = slice_size - overlap
    for i in range(0, len(input_list), step):
        chunk = input_list[i:i + slice_size]
        if chunk:
            chunks.append(chunk)
    return chunks

def chunk_pdf_text(pages_and_texts: List[Dict], num_sentence_chunk_size: int = 20, overlap: int = 9) -> pd.DataFrame:
    all_chunks = []
    for item in pages_and_texts:
        sentences = split_into_sentences(item["text"])
        sentence_chunks = split_list(sentences, slice_size=num_sentence_chunk_size, overlap=overlap)

        for sentence_chunk in sentence_chunks:
            joined_text = " ".join(sentence_chunk)
            joined_text = re.sub(r'\s+', ' ', joined_text)
            chunk_data = {
                "page_number": item["page_number"],
                "sentence_chunk": joined_text,
                "chunk_char_count": len(joined_text),
                "chunk_word_count": len(joined_text.split()),
                "chunk_token_count": len(joined_text) / 4.0  # Rough token approximation
            }
            all_chunks.append(chunk_data)

    return pd.DataFrame(all_chunks)

def embed_chunks(df: pd.DataFrame, model: SentenceTransformer, min_token_length: int = 30) -> List[Dict]:
    filtered = df[df["chunk_token_count"] > min_token_length].to_dict(orient="records")
    for item in tqdm.tqdm(filtered, desc="Generating Embeddings"):
        # The model.encode returns a numpy array, which we store directly in the dictionary
        item["embedding"] = model.encode(item["sentence_chunk"])
    return filtered

def store_in_csv(chunks: List[Dict], csv_file_path: str = "embedding.csv") -> None:
    """
    Stores the list of chunk dictionaries into a CSV file.
    Since embeddings are arrays, we convert them to lists (if needed) and
    store them as strings in the CSV file.
    """
    df = pd.DataFrame(chunks)
    # Convert embedding arrays to lists (or strings) so they can be saved easily to CSV
    df["embedding"] = df["embedding"].apply(lambda x: x.tolist() if hasattr(x, 'tolist') else x)
    df.to_csv(csv_file_path, index=False)
    print(f"Embeddings stored in CSV file '{csv_file_path}'")

def embed_pdf_into_csv(
    pdf_path: str,
    embedding_model_name: str = "dunzhang/stella_en_1.5B_v5",
    num_sentence_chunk_size: int = 20,
    overlap: int = 9,
    min_token_length: int = 30,
    device: str = "cpu",
    csv_file_path: str = "embedding.csv"
) -> None:
    """
    Main pipeline:
    1. Read and clean PDF text.
    2. Chunk text into sentence segments.
    3. Embed chunks using SentenceTransformer model.
    4. Store embeddings in a CSV file.
    """
    if not os.path.exists(pdf_path):
        raise FileNotFoundError(f"File '{pdf_path}' doesn't exist.")

    print(f"Loading embedding model '{embedding_model_name}' on {device}...")
    model = SentenceTransformer(embedding_model_name, device=device, trust_remote_code=True)

    # Process PDF
    pages_and_texts = open_and_read_pdf(pdf_path)
    df_chunks = chunk_pdf_text(pages_and_texts, num_sentence_chunk_size=num_sentence_chunk_size, overlap=overlap)

    # Embed chunks
    chunks = embed_chunks(df_chunks, model=model, min_token_length=min_token_length)

    # Store embeddings in CSV
    store_in_csv(chunks, csv_file_path)

    print(f"PDF embedding process complete. Stored in '{csv_file_path}'.")

if __name__ == "__main__":
    embed_pdf_into_csv(
        pdf_path="./nutrition_handbook.pdf",
        embedding_model_name="sentence-transformers/all-MiniLM-L6-v2",
        num_sentence_chunk_size=20,
        overlap=9,
        min_token_length=30,
        device="cpu",  # Change to 'cuda' if GPU is available
        csv_file_path="embeddings.csv"
    )