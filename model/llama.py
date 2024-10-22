from transformers import pipeline
import torch

messages = [
    {"role": "user", "content": "Who are you?"},
]
pipe = pipeline("text-generation", model="meta-llama/Llama-3.1-8B-Instruct")
# Use max_new_tokens instead of max_length
response = pipe(messages, max_new_tokens=100)
print(response)

from transformers import AutoTokenizer, AutoModelForCausalLM

# Load model and tokenizer
model_name = "meta-llama/Llama-3.1-8B-Instruct"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

# Move model to GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

def generate_response(prompt, max_length=100):
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    
    # Generate
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_length,
            num_return_sequences=1,
            temperature=0.7,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id
        )
    
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return response

# Test the model
messages = [
    {"role": "user", "content": "Who are you?"},
]
prompt = f"Human: {messages[0]['content']}\nAI:"
response = generate_response(prompt)
print("AI:", response)

# Chat loop
print("Chat with the model. Type 'quit' to exit.")
chat_history = ""
while True:
    user_input = input("You: ")
    if user_input.lower() == 'quit':
        break
    
    chat_history += f"Human: {user_input}\n"
    prompt = chat_history + "AI:"
    response = generate_response(prompt)
    chat_history += f"AI: {response}\n"
    print("AI:", response)
