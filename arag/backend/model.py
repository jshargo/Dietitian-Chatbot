import torch
import chromadb
from sentence_transformers import SentenceTransformer
from transformers import AutoModelForCausalLM, AutoTokenizer

# -----------------------------------------
# Global Configuration & One-Time Initialization
# -----------------------------------------

# Default values used by app.py
DEFAULT_CHROMA_DB_PATH = "../data/chromadb"
DEFAULT_EMBEDDING_MODEL_NAME = "dunzhang/stella_en_1.5B_v5"
DEFAULT_LLM_MODEL_NAME = "meta-llama/Llama-3.2-1B-Instruct" 
DEFAULT_N_RESULTS = 3

# Initialize device
device = "cuda" if torch.cuda.is_available() else "cpu"

# Load the embedding model once
embedding_model = SentenceTransformer(DEFAULT_EMBEDDING_MODEL_NAME, device="cpu")

# Connect to Chroma once
chroma_client = chromadb.PersistentClient(path=DEFAULT_CHROMA_DB_PATH)

# Load LLM model and tokenizer once
tokenizer = AutoTokenizer.from_pretrained(DEFAULT_LLM_MODEL_NAME, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(DEFAULT_LLM_MODEL_NAME, device_map="auto", trust_remote_code=True)
model.eval()
model = model.to(device)

# System prompt (adapted from the optimized code)
SYSTEM_PROMPT = """You are a strictly controlled assistant that ONLY uses provided context. Follow these rules WITHOUT EXCEPTION:

1. You must ONLY use information explicitly stated in the provided context
2. If the context doesn't contain the exact information needed, respond ONLY with: "I don't have enough information to answer that question"
3. DO NOT use any external knowledge or common sense, even if you know it's correct
4. DO NOT make assumptions or inferences beyond what's directly stated
5. DO NOT provide partial answers
6. Your response must begin with "Based on the provided context, " followed by your answer
7. If you're unsure if the context fully answers the question, respond with "I don't have enough information to answer that question"

Violation of any of these rules is not acceptable."""

def query_chroma(
    query: str, 
    collection_name: str, 
    chroma_db_path: str = DEFAULT_CHROMA_DB_PATH,  
    embedding_model_name: str = DEFAULT_EMBEDDING_MODEL_NAME,
    n_results: int = DEFAULT_N_RESULTS
) -> dict:
    query_embedding = embedding_model.encode(query).tolist()
    try:
        # Get the requested collection (assuming it exists)
        collection = chroma_client.get_collection(name=collection_name)
        results = collection.query(
            query_embeddings=[query_embedding],
            n_results=n_results
        )
        return results
    except Exception as e:
        print(f"Error querying ChromaDB: {str(e)}")
        return {"documents": [[]]}  


def generate_answer(
    query: str,
    retrieved_results: dict,
    llm_model_name: str = DEFAULT_LLM_MODEL_NAME,
    max_new_tokens: int = 512
) -> str:
    # Extract context from retrieved documents
    documents = retrieved_results.get("documents", [[]])
    context_docs = documents[0] if documents and len(documents) > 0 else []
    
    # If no context is available, return the default response
    if not context_docs:
        return "I don't have enough information to answer that question"
    
    formatted_context = "\n\n".join(context_docs)

    # Add explicit reminder about context limitations
    prompt = f"""{SYSTEM_PROMPT}

Context (ONLY use information from this context):
{formatted_context}

Remember: If the above context doesn't contain the exact information needed to answer the question, respond ONLY with "I don't have enough information to answer that question"

Question: {query}
Answer: """

    # Reduce temperature to make responses more deterministic
    with torch.inference_mode():
        outputs = model.generate(
            **tokenizer(prompt, return_tensors="pt").to(device),
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=0.1,  # Reduced from 0.4 to make responses more consistent
            pad_token_id=tokenizer.eos_token_id
        )

    # Decode and extract answer
    answer = tokenizer.decode(outputs[0], skip_special_tokens=True)
    if "Answer:" in answer:
        answer = answer.split("Answer:", 1)[-1].strip()
    else:
        answer = answer.strip()

    print("Retrieved documents:", documents)
    return answer