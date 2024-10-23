import os
import sys
import fitz  # PyMuPDF
import chromadb
from tqdm import tqdm
from spacy.lang.en import English
import logging
from chromadb.utils import embedding_functions
import tiktoken

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

def create_chunks(text: str, chunk_size=800, overlap=400):
    # Initialize tokenizer
    enc = tiktoken.get_encoding("cl100k_base")
    
    # Tokenize the full text
    tokens = enc.encode(text)
    chunks = []
    
    # Create chunks based on tokens
    for i in range(0, len(tokens), chunk_size - overlap):
        chunk_tokens = tokens[i:i + chunk_size]
        chunk_text = enc.decode(chunk_tokens)
        if len(chunk_tokens) > 30:  # Minimum chunk size filter
            chunks.append(chunk_text)
    return chunks

def main():
    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    # Set up parameters (modify these as needed)
    pdf_path = "nutrition_handbook.pdf"  # Path to your PDF file
    chunk_size = 10
    overlap = 5
    embedding_model_name = 'all-MiniLM-L6-v2'
    collection_name = 'ties_collection'
    embedding_function = embedding_functions.SentenceTransformerEmbeddingFunction(model_name=embedding_model_name)

    # Load and process the document
    document = open_and_read_pdf(pdf_path)

    # Create chunks of sentences
    chunks = create_chunks(document, chunk_size=800, overlap=400)
    logger.info(f"Number of chunks: {len(chunks)}")

    # Initialize ChromaDB client
    path = "vector_db"
    client = chromadb.PersistentClient(path=path)

    # Get or create the collection
    collection = client.get_or_create_collection(
        name=collection_name,
        embedding_function=embedding_function
    )

    # Delete existing documents in the collection (reset the collection)
    existing_ids = collection.get()['ids']
    if existing_ids:
        collection.delete(ids=existing_ids)
        logger.info(f"Deleted existing documents in collection '{collection_name}'.")

    # Prepare data for adding to the collection
    ids = [f"chunk_{i}" for i in range(len(chunks))]
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
