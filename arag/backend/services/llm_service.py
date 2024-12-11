import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

def load_llm_model(path: str):
    tokenizer = AutoTokenizer.from_pretrained(path)
    model = AutoModelForCausalLM.from_pretrained(path, torch_dtype=torch.float32)
    # Optional if torch.compile is supported in your environment:
    # model = torch.compile(model)
    return model, tokenizer

def generate_answer(model, tokenizer, prompt: str, max_length=512):
    inputs = tokenizer(prompt, return_tensors='pt')
    with torch.no_grad():
        outputs = model.generate(**inputs, max_length=max_length)
    return tokenizer.decode(outputs[0], skip_special_tokens=True)
