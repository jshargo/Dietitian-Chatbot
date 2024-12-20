import sys
import torch
import chromadb
from sentence_transformers import SentenceTransformer
from transformers import AutoModelForCausalLM, AutoTokenizer

# ------------------------------
# Configuration & Initialization
# ------------------------------
CHROMA_DB_PATH = "./chromadb"
COLLECTION_NAME = "guidlines_embeddings"
EMBEDDING_MODEL_NAME = "dunzhang/stella_en_1.5B_v5"
LLM_MODEL_NAME = "meta-llama/Llama-3.2-1B-Instruct"

# Device setup
device = "cuda" if torch.cuda.is_available() else "cpu"

# Load embedding model once
embedding_model = SentenceTransformer(EMBEDDING_MODEL_NAME, device="cpu")

# Connect to Chroma once
client = chromadb.PersistentClient(path=CHROMA_DB_PATH)
collection = client.get_collection(name=COLLECTION_NAME)

# Load LLM model and tokenizer once
tokenizer = AutoTokenizer.from_pretrained(LLM_MODEL_NAME, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(LLM_MODEL_NAME, device_map="auto", trust_remote_code=True)
model.eval()
model = model.to(device)

# System prompt for the LLM
SYSTEM_PROMPT = """You are a helpful and professional assistant. Always:
- Provide accurate information based solely on the given context
- Use professional, respectful, and inclusive language
- If the context doesn't contain enough information, honestly say "I don't have enough information to answer that question"
- Keep responses clear and concise
"""


# --------------------
# Helper Functions
# --------------------
def query_documents(query: str, n_results: int = 3):
    # Encode query using the pre-loaded embedding model
    query_embedding = embedding_model.encode(query).tolist()

    # Query the Chroma collection
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=n_results
    )
    return results


def generate_response(query: str, context: list[str], max_length: int = 512) -> str:
    formatted_context = "\n\n".join(context)
    prompt = f"""{SYSTEM_PROMPT}

Context:
{formatted_context}

Question: {query}
Answer: """

    # Tokenize input
    inputs = tokenizer(prompt, return_tensors="pt").to(device)

    # Generate text
    with torch.inference_mode():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_length,
            do_sample=True,
            temperature=0.1,
            top_p=0.9,
            pad_token_id=tokenizer.eos_token_id
        )

    # Decode and extract answer
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    if "Answer:" in response:
        response = response.split("Answer:", 1)[-1].strip()
    return response.strip()


# --------------------
# Main Execution Block
# --------------------
if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python run_query.py <query>")
        sys.exit(1)

    user_query = sys.argv[1]
    results = query_documents(user_query)
    if not results or not results['documents'] or len(results['documents'][0]) == 0:
        print("I don't have enough information to answer that question")
        sys.exit(0)

    # Get the top retrieved documents
    context_docs = results['documents'][0]
    response = generate_response(user_query, context_docs)
    print(response)