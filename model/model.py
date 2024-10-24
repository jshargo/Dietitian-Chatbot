# For AI server since sqlite is not installed and python version is 3.8
import sys
__import__('pysqlite3')
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')

import os
import chromadb
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from chromadb.utils import embedding_functions
from typing import List
import logging

# Logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def prompt_formatter(query: str, context_documents: List[str]) -> str:
    context = "\n\n".join([f"[{i+1}] {doc}" for i, doc in enumerate(context_documents)])
    system_prompt = "You are a helpful assistant."
    prompt = f"""### System:
{system_prompt}

### Context:
{context}

### User:
{query}

### Assistant:"""
    return prompt

def rag_query(query: str, collection, tokenizer, llm_model, device, temperature=0.8, max_new_tokens=200, n_results=5):
    try:
        # Retrieve 
        results = collection.query(query_texts=[query], n_results=n_results)
        documents = results['documents'][0]

        # Format
        prompt = prompt_formatter(query, documents)

        # Tokenize 
        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=2048).to(device)

        # Generate response
        with torch.no_grad():
            outputs = llm_model.generate(
                input_ids=inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
                temperature=temperature,
                do_sample=True,
                max_new_tokens=max_new_tokens,
                top_p=0.95,
                top_k=50,
                repetition_penalty=1.2,
                eos_token_id=tokenizer.eos_token_id,
                pad_token_id=tokenizer.pad_token_id
            )

        # Decode and format
        output_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        # Extract reply
        answer = output_text[len(prompt):].strip()
        return answer

    except Exception as e:
        logger.error(f"An error occurred during query: {e}")
        return f"An error occurred: {str(e)}. Please try again or rephrase your question."

def main():
    # Get query from CLI
    if len(sys.argv) < 2:
        print("Please provide a query as a command-line argument.")
        sys.exit(1)
    query = sys.argv[1]

    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    # Set up parameters 
    model_id = 'meta-llama/Llama-3.1-8B-Instruct'  # model id
    embedding_model_name = 'sentence-transformers/all-MiniLM-L6-v2'  # embedding model
    collection_name = 'ties_collection'  # Chromadb
    n_results = 5
    temperature = 0.8
    max_new_tokens = 200
    path = "vector_db"  # chroma path

    # Device 
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if device.type == 'cpu' and torch.backends.mps.is_available():
        device = torch.device('mps')

    logger.info(f"Using device: {device}")

    # Init ChromaDB 
    client = chromadb.PersistentClient(path=path)

    # Init embedding function
    embedding_function = embedding_functions.SentenceTransformerEmbeddingFunction(model_name=embedding_model_name)

    # Get or create collection
    collection = client.get_or_create_collection(
        name=collection_name,
        embedding_function=embedding_function
    )

    # Check collection 
    if collection.count() == 0:
        logger.error("The ChromaDB collection is empty. Please run index_documents.py first to index documents.")
        sys.exit(1)

    # Init LLM model
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        llm_model = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype=torch.float16 if device.type != 'cpu' else torch.float32,
            low_cpu_mem_usage=True
        )
        llm_model.to(device)
        llm_model.eval()
    except Exception as e:
        logger.error(f"Error loading LLM model: {e}")
        sys.exit(1)

    response = rag_query(
        query=query,
        collection=collection,
        tokenizer=tokenizer,
        llm_model=llm_model,
        device=device,
        temperature=temperature,
        max_new_tokens=max_new_tokens,
        n_results=n_results
    )

    print("\n" + "="*50)
    print(f"Query: {query}")
    print(f"Response: {response}")
    print("="*50 + "\n")

if __name__ == '__main__':
    main()
