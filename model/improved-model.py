import os
import numpy as np
import pandas as pd
import torch
from typing import List, Dict
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig


os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
if device.type == 'cpu' and torch.backends.mps.is_available():
    device = torch.device('mps')

print(f"Using device: {device}")


# Init Model
model_id = "nvidia/Mistral-NeMo-Minitron-8B-Instruct"
config = AutoConfig.from_pretrained(pretrained_model_name_or_path=model_id, hidden_activation="gelu_pytorch_tanh", token=True)
tokenizer = AutoTokenizer.from_pretrained(
    pretrained_model_name_or_path=model_id, 
    token=True,
    clean_up_tokenization_spaces=False
)
llm_model = AutoModelForCausalLM.from_pretrained(pretrained_model_name_or_path=model_id, 
                                                 config=config, 
                                                 torch_dtype=torch.float16, 
                                                 low_cpu_mem_usage=False,
                                                 token=True)
llm_model.to(device)

embedding_model = SentenceTransformer("all-mpnet-base-v2").to(device)

# Load embeddings
csv_path = "dataset_folder/text_chunks_and_embeddings_df.csv"
text_chunks_and_embedding_df = pd.read_csv(csv_path)

text_chunks_and_embedding_df["embedding"] = text_chunks_and_embedding_df["embedding"].apply(lambda x: np.fromstring(x.strip("[]"), sep=" ")) # For similarity calculation
pages_and_chunks = text_chunks_and_embedding_df.to_dict(orient="records")

embeddings = np.array(text_chunks_and_embedding_df["embedding"].tolist()).astype('float32')
dimension = embeddings.shape[1]

def retrieve_relevant_resources(query: str, n_resources_to_return: int = 5) -> List[Dict]:
    query_embedding = embedding_model.encode(query, convert_to_tensor=True, device=device)
    
    # Calculate cosine similarity between query and all embeddings
    similarities = np.dot(embeddings, query_embedding.cpu().numpy())
    
    # Get top k indices
    top_k_indices = np.argsort(similarities)[-n_resources_to_return:][::-1]
    top_k_scores = similarities[top_k_indices]
    
    return [{"chunk": pages_and_chunks[i], "score": float(score)} 
            for i, score in zip(top_k_indices, top_k_scores)]

def prompt_formatter(query: str, context_items: List[Dict]) -> str:
    context = "\n".join([f"[{i+1}] {item['chunk']['sentence_chunk']}" for i, item in enumerate(context_items)])
    base_prompt = f"""Answer the following query based on the provided context. If the answer is not in the context, please state that you don't have enough information.

Context:
{context}

Query: {query}

Answer:
"""
    return base_prompt

def ask(query: str, temperature=0.3, max_new_tokens=256) -> str:
    try:
        context_items = retrieve_relevant_resources(query)
        prompt = prompt_formatter(query, context_items)
        inputs = tokenizer(prompt, return_tensors="pt").to(device)
        
        with torch.no_grad():
            outputs = llm_model.generate(
                **inputs,
                temperature=temperature,
                do_sample=True,
                max_new_tokens=max_new_tokens,
                top_k=50,
                eos_token_id=tokenizer.eos_token_id,
                repetition_penalty=1.2,
            )
        
        output_text = tokenizer.decode(outputs[0], skip_special_tokens=True, clean_up_tokenization_spaces=False)
        answer = output_text.split("Answer:")[-1].strip()
        answer = answer.split("Answer:")[0].strip()  # Remove any repeated 'Answer:' sections
        return answer
    except Exception as e:
        return f"An error occurred: {str(e)}. Please try again or rephrase your question."


if __name__ == "__main__":
    while True:
        query = input("Enter your question: ")
        response = ask(query)
        print(f"Answer: {response}\n")
