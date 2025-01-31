import logging
import uvicorn
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from dotenv import load_dotenv

from model2 import Model
from potts import IntentClassifier

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

load_dotenv()
app = FastAPI()
classifier = IntentClassifier()
model= Model() 

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
    '''
    Can also recieve:   "user_id": user_id,
                        "context": {
                            "user_profile": {
                                "age": user.age,
                                "sex": user.sex,
                                "height": user.height,
                                "weight": user.weight,
                                "activity_level": user.activity_level
                            },
    '''
    
    intent = classifier.classify(query)
    top_intent = intent["top_intent"]
    top_category = intent["top_category"]
    
    if not query:
        return JSONResponse({"error": "No query provided"}, status_code=400)
    
    logger.info(f"Received query: {query}")
    
    response = model.get_response(query)
    
    return JSONResponse({
        "reasoning": response["reasoning"],
        "final_answer": response
    })


@app.get("/health")
async def health_check():
    """Simple endpoint to confirm the app is running."""
    return {"status": "healthy"}

if __name__ == "__main__":
    # Run the FastAPI server
    uvicorn.run("main:app", host="0.0.0.0", port=8001, reload=True)