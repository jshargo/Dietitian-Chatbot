import os
import numpy as np
import pandas as pd
import torch
from typing import List
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig
import faiss
import warnings

warnings.filterwarnings("ignore", category=FutureWarning, module="transformers.tokenization_utils_base")

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
if device.type == 'cpu' and torch.backends.mps.is_available():
    device = torch.device('mps')

print(f"Using device: {device}")

# Init model
model_id = "google/gemma-2b-it"
config = AutoConfig.from_pretrained(pretrained_model_name_or_path=model_id, hidden_activation="gelu_pytorch_tanh", token=True)
tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name_or_path=model_id, token=True)
llm_model = AutoModelForCausalLM.from_pretrained(pretrained_model_name_or_path=model_id, 
                                                 config=config, 
                                                 torch_dtype=torch.float16, 
                                                 low_cpu_mem_usage=False,
                                                 token=True)
llm_model.to(device)

embedding_model = SentenceTransformer("all-mpnet-base-v2").to(device)

csv_path = "dataset_folder/text_chunks_and_embeddings_df.csv"

def load_data(csv_path: str):
    global index  
    df = pd.read_csv(csv_path)
    
    if "embedding" not in df.columns:
        raise ValueError("The CSV file must contain an 'embedding' column.")
    
    df["embedding"] = df["embedding"].apply(lambda x: np.fromstring(x.strip("[]"), sep=" "))
    
    embeddings = np.array(df["embedding"].tolist()).astype('float32')
    index = faiss.IndexFlatIP(embeddings.shape[1])
    if device.type == 'cuda':
        res = faiss.StandardGpuResources()
        index = faiss.index_cpu_to_gpu(res, 0, index)
    index.add(embeddings)
    
    return df

def retrieve_relevant_embeddings(query: str, n_results: int = 5) -> List[np.ndarray]:
    query_embedding = embedding_model.encode(query, convert_to_tensor=True, device=device)
    scores, indices = index.search(query_embedding.cpu().numpy().reshape(1, -1), n_results)
    return [embeddings[i] for i in indices[0]]

def reconstruct_text_from_embedding(embedding: np.ndarray) -> str:
    return f"[Content represented by embedding {hash(embedding.tobytes())%1000}]"

def prompt_formatter(query: str, relevant_embeddings: List[np.ndarray]) -> str:
    context = "\n\n".join([f"[{i+1}] {reconstruct_text_from_embedding(emb)}" for i, emb in enumerate(relevant_embeddings)])
    base_prompt = """You are a friendly and knowledgeable dietary support assistant. Your goal is to provide helpful, accurate, and easy-to-understand information about nutrition and diet. Always respond in a warm and supportive manner, as if you're talking to a friend. Here are some guidelines for your responses:

1. When you know the answer:
   Start with an enthusiastic acknowledgment, then provide the information clearly and concisely. For example:
   "Absolutely! I'd be happy to help with that." or "Great question! Here's what you need to know:"

2. When you're not certain or don't have enough information:
   Be honest about your limitations and offer to provide related information if possible. For example:
   "I'm not entirely sure about that specific detail, but here's what I do know that might be helpful:" or
   "While I don't have exact information on that, I can share some related insights that you might find useful:"

3. When you don't have the information:
   Politely explain that you don't have the answer, and if appropriate, suggest where they might find more information. For example:
   "I apologize, but I don't have specific information about that in my knowledge base. This might be something to discuss with a registered dietitian or your healthcare provider for the most accurate advice."

4. Providing advice:
   Always emphasize that your information is general and not a substitute for professional medical advice. For example:
   "Based on general nutritional guidelines, ... However, remember that everyone's dietary needs are unique, so it's always best to consult with a healthcare professional or registered dietitian for personalized advice."

5. Encouraging healthy habits:
   Whenever appropriate, gently encourage balanced eating, regular physical activity, and overall wellness.

6. Explaining complex topics:
   Break down complex nutritional concepts into simple, easy-to-understand language. Use analogies or everyday examples when possible to make the information more relatable.

7. Empathy and support:
   Show empathy and understanding, especially when discussing sensitive topics like weight management or dietary restrictions.

Remember, your role is to be a supportive guide in the user's nutrition journey. Provide information that empowers them to make informed decisions about their diet and overall health.

Based on the following context items derived from embeddings of a nutrition textbook, please answer the query. If the information is not available in the context, please state that you don't have enough information to answer accurately, but offer general advice if possible.

Context:
{context}

Query: {query}

Answer: Let's approach this step-by-step:

1) First, I'll identify the key points in the query.
2) Then, I'll analyze the provided context derived from relevant embeddings.
3) Finally, I'll synthesize this information to provide a comprehensive, friendly, and supportive answer related to nutrition.

Here's my response:
"""
    return base_prompt.format(context=context, query=query)


def ask(query: str, temperature = 0.7, max_new_tokens = 512) -> str:
    try:
        relevant_embeddings = retrieve_relevant_embeddings(query)
        prompt = prompt_formatter(query, relevant_embeddings)
        inputs = tokenizer(prompt, return_tensors="pt").to(device)
        
        with torch.no_grad():
            outputs = llm_model.generate(
                **inputs,
                temperature=temperature,
                do_sample=True,
                max_new_tokens=max_new_tokens,
                top_p=0.95,
                top_k=50,
                repetition_penalty=1.2
            )
        
        output_text = tokenizer.decode(outputs[0], skip_special_tokens=True, clean_up_tokenization_spaces=True)
        answer = output_text.split("Here's my response:")[-1].strip()
        return answer
    except Exception as e:
        return f"An error occurred: {str(e)}. Please try again or rephrase your question."

if __name__ == "__main__":
    try:
        print("Loading and processing data...")
        df = load_data(csv_path)
        embeddings = np.array(df["embedding"].tolist()).astype('float32')
        print("Data loaded and processed. Ready for questions.")
        
        while True:
            query = input("Enter your question: ")
            if query.lower() == 'quit':
                break
            response = ask(query)
            print(f"Answer: {response}\n")
    except Exception as e:
        print(f"An error occurred: {str(e)}")
        print("Please check your CSV file and ensure it contains the necessary 'embedding' column.")