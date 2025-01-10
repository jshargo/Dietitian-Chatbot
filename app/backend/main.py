import os
import uvicorn
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
import logging
from potts import IntentClassifier

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

app = FastAPI()
intent_classifier = IntentClassifier()
logger = logging.getLogger(__name__)

@app.post("/api/query")
async def handle_user_query(payload: dict):
    """
    Entry point for user queries from the frontend.
    """
    try:
        logger.info(f"Received query payload: {payload}")
        query_text = payload.get("query", "")
        
        # Use the IntentClassifier to classify the query
        result = intent_classifier.classify(query_text)
        
        return JSONResponse({
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