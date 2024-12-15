import chromadb
from sentence_transformers import SentenceTransformer
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline as hf_pipeline

def query_chroma(query: str, collection_name: str, chroma_db_path: str = "./chroma_db", embedding_model_name: str = "all-mpnet-base-v2", n_results: int = 3) -> dict:
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
    context = "\n\n".join(retrieved_results["documents"][0]) if "documents" in retrieved_results and retrieved_results["documents"] else ""
    prompt = f"Answer the following query based on the given context.\n\nContext:\n{context}\n\nQuery: {query}\nAnswer: "

    tokenizer = AutoTokenizer.from_pretrained(llm_model_name)
    model = AutoModelForCausalLM.from_pretrained(llm_model_name)
    gen_pipeline = hf_pipeline("text-generation", model=model, tokenizer=tokenizer, device_map="auto")

    output = gen_pipeline(prompt, max_length=max_length, do_sample=False, top_p=0.9)[0]["generated_text"]
    answer = output.split("Answer:")[-1].strip()
    return answer

if __name__ == "__main__":
    # Run this script to query and generate an answer
    collection_name = "pdf_embeddings"
    embedding_model_name = "all-mpnet-base-v2"
    chroma_db_path = "./chroma_db"
    user_query = "What dietary recommendations are given for managing diabetes?"

    results = query_chroma(query=user_query, collection_name=collection_name, chroma_db_path=chroma_db_path, embedding_model_name=embedding_model_name, n_results=3)
    print("Retrieved results:", results)

    final_answer = generate_answer(query=user_query, retrieved_results=results, llm_model_name="meta-llama/Llama-3.2-1B")
    print("Final Answer:", final_answer)