import os
import sys
import fitz  # PyMuPDF
import chromadb
from chromadb.config import Settings
from tqdm import tqdm
import logging
from chromadb.utils import embedding_functions
from transformers import AutoTokenizer
import uuid

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def text_formatter(text: str) -> str:
    return text.replace("\n", " ").strip()

def open_and_read_pdf(pdf_path: str) -> str:
    try:
        doc = fitz.open(pdf_path)
        full_text = ""
        for page in tqdm(doc, desc="Reading PDF"):
            text = page.get_text()
            full_text += text_formatter(text) + " "
        return full_text.strip()
    except Exception as e:
        logger.error(f"Error reading PDF: {e}")
        sys.exit(1)

def create_chunks(text: str, tokenizer, max_tokens=512, overlap_tokens=50):
    tokens = tokenizer.encode(text)
    chunks = []
    start = 0
    end = 0
    text_length = len(tokens)
    while start < text_length:
        end = min(start + max_tokens, text_length)
        chunk_tokens = tokens[start:end]
        chunk_text = tokenizer.decode(chunk_tokens, skip_special_tokens=True)
        chunks.append(chunk_text)
        start += max_tokens - overlap_tokens
    return chunks

def main():
    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    # Set up parameters (modify these as needed)
    pdf_path = "nutrition_handbook.pdf"  # Path to your PDF file
    max_tokens = 512  # Maximum tokens per chunk
    overlap_tokens = 50  # Overlap between chunks
    embedding_model_name = 'sentence-transformers/all-MiniLM-L6-v2'
    collection_name = 'ties_collection'
    embedding_function = embedding_functions.SentenceTransformerEmbeddingFunction(model_name=embedding_model_name)

    # Load and process the document
    document = open_and_read_pdf(pdf_path)

    # Initialize tokenizer
    tokenizer = AutoTokenizer.from_pretrained(embedding_model_name)

    # Create chunks of sentences
    chunks = create_chunks(document, tokenizer, max_tokens=max_tokens, overlap_tokens=overlap_tokens)
    logger.info(f"Number of chunks: {len(chunks)}")

    # Initialize ChromaDB client with new configuration
    path = "vector_db"
    client = chromadb.Client(Settings(
        persist_directory=path  # Specify the directory for persistence
    ))

    # Get or create the collection
    collection = client.get_or_create_collection(
        name=collection_name,
        embedding_function=embedding_function
    )

    # Optional: Delete existing documents from the specific PDF
    results = collection.get(
        where={"source": pdf_path}
    )
    existing_ids = results['ids']
    if existing_ids:
        collection.delete(ids=existing_ids)
        logger.info(f"Deleted existing documents from '{pdf_path}' in collection '{collection_name}'.")

    # Prepare data for adding to the collection
    ids = [f"{os.path.basename(pdf_path)}_chunk_{i}_{uuid.uuid4()}" for i in range(len(chunks))]
    metadatas = [{'source': pdf_path} for _ in range(len(chunks))]

    # Add documents to collection
    logger.info("Adding documents to ChromaDB collection...")
    collection.add(
        documents=chunks,
        metadatas=metadatas,
        ids=ids
    )

    logger.info("Indexing completed successfully.")

if __name__ == '__main__':
    main()
