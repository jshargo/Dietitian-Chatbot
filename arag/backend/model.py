import chromadb
from sentence_transformers import SentenceTransformer
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline as hf_pipeline

def query_chroma(query: str, collection_name: str, chroma_db_path: str = "./data/chroma_db", embedding_model_name: str = "all-mpnet-base-v2", n_results: int = 3) -> dict:
    embedding_model = SentenceTransformer(embedding_model_name, device="cpu")
    query_embedding = embedding_model.encode(query).tolist()

    chroma_client = chromadb.PersistentClient(path=chroma_db_path)
    collection = chroma_client.get_collection(name=collection_name)

    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=n_results
    )
    return results

def generate_answer(query: str, retrieved_results: dict, llm_model_name: str = "meta-llama/Llama-3.2-1B", max_length: int = 512) -> str:
    context = ""
    if "documents" in retrieved_results and retrieved_results["documents"]:
        # Join all retrieved docs for context
        for doc_list in retrieved_results["documents"]:
            for doc in doc_list:
                context += doc + "\n\n"

    prompt = f"""You are a helpful RAG assistant.
When the user asks you a question, answer only based on the given context.
If you can't find an answer based on the context, reply "I don't know".

Context:
{context}

Query: {query}
Answer:"""

    tokenizer = AutoTokenizer.from_pretrained(llm_model_name)
    model = AutoModelForCausalLM.from_pretrained(llm_model_name)
    gen_pipeline = hf_pipeline("text-generation", model=model, tokenizer=tokenizer, device_map="auto")

    output = gen_pipeline(prompt, max_length=max_length, do_sample=True, temperature=0.2)[0]["generated_text"]
    # Extract answer after "Answer:"
    answer = output.split("Answer:")[-1].strip()
    return answer
