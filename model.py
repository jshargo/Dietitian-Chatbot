import numpy as np
import pandas as pd
import textwrap
import torch
from sentence_transformers import util, SentenceTransformer
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig
import sys

import warnings
warnings.filterwarnings("ignore", message="`clean_up_tokenization_spaces` was not set")

query = sys.argv[1]

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

csv_path = "text_chunks_and_embeddings_df.csv"  
text_chunks_and_embedding_df = pd.read_csv(csv_path)

text_chunks_and_embedding_df["embedding"] = text_chunks_and_embedding_df["embedding"].apply(lambda x: np.fromstring(x.strip("[]"), sep=" "))
pages_and_chunks = text_chunks_and_embedding_df.to_dict(orient="records")
embeddings = torch.tensor(np.array(text_chunks_and_embedding_df["embedding"].tolist()), dtype=torch.float32).to(device)

embedding_model = SentenceTransformer(model_name_or_path="all-mpnet-base-v2", device=device)

def retrieve_relevant_resources(query: str, embeddings: torch.tensor, model: SentenceTransformer=embedding_model, n_resources_to_return: int=5):
    query_embedding = model.encode(query, convert_to_tensor=True, device=device) 
    dot_scores = util.dot_score(query_embedding, embeddings)[0]
    scores, indices = torch.topk(input=dot_scores, k=n_resources_to_return)
    return scores, indices

def print_wrapped(text, wrap_length=80):
    wrapped_text = textwrap.fill(text, wrap_length)
    print(wrapped_text)

def prompt_formatter(query: str, context_items: list[dict]) -> str:
    context = "- " + "\n- ".join([item["sentence_chunk"] for item in context_items])
    base_prompt = """Based on the following context items, please answer the query.
Give yourself room to think by extracting relevant passages from the context before answering the query.
Don't return the thinking, only return the answer.
Make sure your answers are as in-depth as possible.
Use the following examples as reference for the ideal answer style.
\nExample 1:
Query: What is a healthy breakfast option?
Answer: A healthy breakfast is one that provides a good balance of macronutrients (carbohydrates, proteins, and fats), essential vitamins and minerals, and fiber, which together help sustain energy levels, support metabolism, and promote overall well-being. One example of this is oatmeal topped with berries, a spoonful of almond butter, and a side of Greek yogurt.
\nExample 2:
Query: What are the causes of type 2 diabetes and how can prevent it?
Answer: Type 2 diabetes is primarily caused by a combination of genetic factors and lifestyle choices, such as obesity, lack of physical activity, poor diet, and insulin resistance. It can be prevented by maintaining a healthy weight, eating a balanced diet rich in whole foods, staying physically active, managing stress, and avoiding excessive consumption of sugary and processed foods.
\nExample 3:
Query: What foods should I look at if I want to lose weight and improve heart health?
Answer: Focus on whole foods like vegetables, fruits, whole grains, lean proteins (such as fish, chicken, and legumes), and healthy fats (like avocados, nuts, and olive oil). Also, include foods high in fiber, like oats and leafy greens, and limit processed foods, sugary snacks, and saturated fats to support weight loss and heart health.
\nNow use the following context items to answer the user query:
{context}
\nRelevant passages: <extract relevant passages from the context here>
User query: {query}
Answer:"""
    base_prompt = base_prompt.format(context=context, query=query)
    dialogue_template = [{"role": "user", "content": base_prompt}]
    prompt = tokenizer.apply_chat_template(conversation=dialogue_template, tokenize=False, add_generation_prompt=True)
    return prompt

def ask(query, temperature=0.1, max_new_tokens=512, format_answer_text=True, return_answer_only=True):
    scores, indices = retrieve_relevant_resources(query=query, embeddings=embeddings)
    context_items = [pages_and_chunks[i] for i in indices]
    for i, item in enumerate(context_items):
        item["score"] = scores[i].cpu()
    prompt = prompt_formatter(query=query, context_items=context_items)
    input_ids = tokenizer(prompt, return_tensors="pt").to(device)
    outputs = llm_model.generate(**input_ids, temperature=temperature, do_sample=True, max_new_tokens=max_new_tokens)
    output_text = tokenizer.decode(outputs[0])

    if format_answer_text:
        #clean up by removing the prompt and unnecessary info
        output_text = output_text.replace(prompt, "").strip()
        output_text = output_text.replace("<bos>", "").replace("<eos>", "").strip()

    if return_answer_only:
        return output_text

    return output_text, context_items


# init the LLM model
model_id = "google/gemma-2b-it"
config = AutoConfig.from_pretrained(pretrained_model_name_or_path=model_id, hidden_activation="gelu_pytorch_tanh")
tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name_or_path=model_id)
llm_model = AutoModelForCausalLM.from_pretrained(pretrained_model_name_or_path=model_id, 
                                                 config=config, 
                                                 torch_dtype=torch.float16, 
                                                 low_cpu_mem_usage=False)
llm_model.to(device)

answer = ask(query=query)
print(f"Query: {query}")
print(f"Answer:\n{answer}")