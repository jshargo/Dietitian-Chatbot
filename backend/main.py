# main.py
import logging
import uvicorn
import os
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from dotenv import load_dotenv

from model import Model
from potts import IntentClassifier

os.environ["TOKENIZERS_PARALLELISM"] = "false"

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

load_dotenv()
app = FastAPI()
classifier = IntentClassifier()
model = Model() 

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], 
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post("/query")
async def process_query(request: Request):
    
    data = await request.json()
    query = data.get("query") or data.get("msg")
    user_context = data.get("context", {}).get("user_profile", {})
    
    if not query:
        return JSONResponse({"error": "No query provided"}, status_code=400)

    logger.info(f"Received query: {query}")
    logger.info(f"User context: {user_context}")

    # Pass all user context to the model.
    response = model.get_response(query, user_context=user_context)

    return JSONResponse({
        "reasoning": response["reasoning"],
        "final_answer": response["final_answer"],
        "detected_intent": response["detected_intent"],
        "context_used": response.get("context_used", ""),
        "raw_content": response.get("raw_content", "")
    })

@app.get("/health")
async def health_check():
    """Simple endpoint to confirm the app is running."""
    return {"status": "healthy"}

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8001, reload=True)