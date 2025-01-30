import os
import uvicorn
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import logging
from potts import IntentClassifier
from agents import MealLoggingAgent, MealPlanningAgent, EducationalAgent, PersonalizedAdviceAgent
from database import get_db_session

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

app = FastAPI()
intent_classifier = IntentClassifier()
logger = logging.getLogger(__name__)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, replace with your frontend URL
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post("/api/query")
async def process_query(request: Request):
    try:
        data = await request.json()
        query = data.get("query") or data.get("msg")  # Handle both frontend formats
        
        if not query:
            return JSONResponse({"error": "No query provided"}, status_code=400)
            
        logger.info(f"Received query: {query}")
        
        # Get intent classification
        result = intent_classifier.classify(query)
        logger.info(f"Classification result: {result}")
        
        return JSONResponse({
            "response": f"I understand you want help with: {result['top_intent']}",
            "intent": result["top_intent"],
            "category": result["top_category"],
            "classifications": result["classifications"]
        })
    except Exception as e:
        logger.error(f"Error processing query: {str(e)}", exc_info=True)
        return JSONResponse({"error": str(e)}, status_code=500)

@app.get("/health")
async def health_check():
    return {"status": "healthy"}


if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8001)