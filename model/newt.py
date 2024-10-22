import os
import numpy as np
import pandas as pd
import torch
from typing import List, Dict
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig
import faiss
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)

class RAGChatbot:
    def __init__(self, model_id: str, embedding_model_id: str):
        # Device configuration
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        if self.device.type == 'cpu' and torch.backends.mps.is_available():
            self.device = torch.device('mps')
        logging.info(f"Using device: {self.device}")

        # Load LLM model
        self.model_id = model_id
        self.config = AutoConfig.from_pretrained(
            pretrained_model_name_or_path=self.model_id,
            hidden_activation="gelu_new",
        )
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_id)
        self.llm_model = AutoModelForCausalLM.from_pretrained(
            self.model_id,
            config=self.config,
            torch_dtype=torch.float16 if self.device.type == 'cuda' else torch.float32,
            low_cpu_mem_usage=True,
        ).to(self.device)

        # Load embedding model
        self.embedding_model = SentenceTransformer(embedding_model_id).to(self.device)

        # Load data and embeddings
        self.pages_and_chunks = self.load_data()

        # Load or create FAISS index
        self.index = self.load_faiss_index()

    def load_data(self):
        # Load data
        csv_path = "dataset_folder/text_chunks_and_embeddings_df.csv"
        if not os.path.exists(csv_path):
            raise FileNotFoundError(f"Data file not found at {csv_path}")

        logging.info("Loading data...")
        text_chunks_and_embedding_df = pd.read_csv(csv_path)

        # Load embeddings from a binary file
        embeddings_path = "dataset_folder/embeddings.npy"
        if os.path.exists(embeddings_path):
            embeddings = np.load(embeddings_path)
        else:
            # Convert embeddings from string to numpy array and save as binary file
            logging.info("Processing embeddings...")
            text_chunks_and_embedding_df["embedding"] = text_chunks_and_embedding_df["embedding"].apply(
                lambda x: np.fromstring(x.strip("[]"), sep=" ")
            )
            embeddings = np.vstack(text_chunks_and_embedding_df["embedding"].values)
            np.save(embeddings_path, embeddings)

        self.embeddings = embeddings.astype('float32')
        # Store pages and chunks
        pages_and_chunks = text_chunks_and_embedding_df.drop(columns=['embedding']).to_dict(orient="records")

        return pages_and_chunks

    def load_faiss_index(self):
        index_path = "faiss_index.bin"
        dimension = self.embeddings.shape[1]

        if os.path.exists(index_path):
            logging.info("Loading existing FAISS index...")
            index = faiss.read_index(index_path)
        else:
            logging.info("Creating new FAISS index...")
            index = faiss.IndexFlatIP(dimension)
            index.add(self.embeddings)
            faiss.write_index(index, index_path)

        # Move index to GPU if available
        if faiss.get_num_gpus() > 0:
            logging.info("Moving FAISS index to GPU...")
            index = faiss.index_cpu_to_all_gpus(index)

        logging.info(f"FAISS index contains {index.ntotal} vectors")
        return index

    def sanitize_input(self, user_input: str) -> str:
        # Simple sanitization to prevent prompt injection
        return user_input.replace("<", "&lt;").replace(">", "&gt;")

    def retrieve_relevant_resources(self, query: str, n_resources_to_return: int = 5) -> List[Dict]:
        query_embedding = self.embedding_model.encode(query, convert_to_tensor=True).to(self.device)
        query_embedding_np = query_embedding.detach().cpu().numpy().astype('float32')

        scores, indices = self.index.search(query_embedding_np.reshape(1, -1), n_resources_to_return)
        results = []
        for idx, score in zip(indices[0], scores[0]):
            item = {"chunk": self.pages_and_chunks[idx], "score": float(score)}
            results.append(item)
        return results

    def prompt_formatter(self, query: str, context_items: List[Dict]) -> str:
        # Limit the context to fit within the model's maximum input length
        max_context_length = 2048  # Adjust based on model's capacity
        context = ""
        for i, item in enumerate(context_items):
            text = f"[{i+1}] {item['chunk']['sentence_chunk']}"
            if len(self.tokenizer.encode(context + text)) < max_context_length:
                context += text + "\n"
            else:
                break

        # Base prompt
        base_prompt = f"""You are an AI assistant that provides helpful answers to the user's queries based on the provided context. If the information is not available in the context, politely inform the user that you don't have enough information.

Context:
{context}

Query: {query}

Answer:"""
        return base_prompt

    def ask(self, query: str, temperature: float = 0.7, max_new_tokens: int = 512) -> str:
        try:
            sanitized_query = self.sanitize_input(query)
            context_items = self.retrieve_relevant_resources(sanitized_query)
            prompt = self.prompt_formatter(sanitized_query, context_items)
            inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)

            # Ensure input length does not exceed model's max length
            max_input_length = self.llm_model.config.max_position_embeddings
            if inputs['input_ids'].size(1) > max_input_length:
                inputs = {k: v[:, -max_input_length:] for k, v in inputs.items()}

            # Generate response
            with torch.no_grad():
                outputs = self.llm_model.generate(
                    **inputs,
                    temperature=temperature,
                    do_sample=True,
                    max_new_tokens=max_new_tokens,
                    top_p=0.9,
                    top_k=40,
                    repetition_penalty=1.1,
                    eos_token_id=self.tokenizer.eos_token_id,
                )

            output_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            answer = output_text.split("Answer:")[-1].strip()
            return answer
        except Exception as e:
            logging.error("Error in 'ask' function", exc_info=True)
            return "An internal error occurred. Please try again later."

if __name__ == "__main__":
    model_id = "meta-llama/Llama-3.1-8B-Instruct"  # Update to your desired model
    embedding_model_id = "all-mpnet-base-v2"

    chatbot = RAGChatbot(model_id, embedding_model_id)
    while True:
        query = input("Enter your question: ")
        response = chatbot.ask(query)
        print(f"Answer: {response}\n")
