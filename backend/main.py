# backend/main.py
from fastapi import FastAPI, Request
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
from database import SessionLocal
from models import User, Agent, Conversation, Message
from cache import get_chat_history, update_chat_history
from vector_db import search_vector_db
from typing import AsyncGenerator
from starlette.responses import Response

app = FastAPI()

# CORS settings
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  #
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load the model and tokenizer
tokenizer = AutoTokenizer.from_pretrained("nvidia/Mistral-NeMo-Minitron-8B-Instruct")
model = AutoModelForCausalLM.from_pretrained("nvidia/Mistral-NeMo-Minitron-8B-Instruct")
model.eval()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

@app.get("/chat")
async def chat(request: Request, message: str, user_id: int, agent_id: int):
    # Fetch chat history from Redis cache
    chat_history = get_chat_history(user_id, agent_id)

    # Update chat history with the new user message
    chat_history.append({"role": "user", "content": message})
    update_chat_history(user_id, agent_id, chat_history)

    # Search VectorDB (Chroma) for relevant context
    context = search_vector_db(message)

    # Prepare input for the model
    input_text = ""
    for msg in chat_history:
        input_text += f"{msg['role']}: {msg['content']}\n"
    input_text += f"assistant:"

    # Tokenize input
    inputs = tokenizer.encode(input_text, return_tensors='pt').to(device)

    # Streaming generator
    async def event_generator() -> AsyncGenerator[str, None]:
        output_ids = inputs
        past_key_values = None

        for _ in range(200):  # Max new tokens
            outputs = model(input_ids=output_ids, past_key_values=past_key_values, use_cache=True)
            logits = outputs.logits
            past_key_values = outputs.past_key_values

            next_token_logits = logits[:, -1, :]
            next_token_id = torch.argmax(next_token_logits, dim=-1, keepdim=True)
            output_ids = next_token_id

            generated_token = tokenizer.decode(next_token_id.squeeze(), skip_special_tokens=True)

            if generated_token:
                yield f"data: {generated_token}\n\n"

            # Stop if end of sentence token is generated
            if next_token_id.item() == tokenizer.eos_token_id:
                break

    return StreamingResponse(event_generator(), media_type="text/event-stream")

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000)
