import os
import uvicorn
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import logging
from potts import IntentClassifier
from agents import MealLoggingAgent, MealPlanningAgent, EducationalAgent, PersonalizedAdviceAgent
from database import get_db_session
from model import RAGEngine
from database import UserProfile

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
    allow_origins=["*"],  
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post("/api/query")
async def process_query(request: Request):
    # Parse incoming data
    data = await request.json()
    query = data.get("query", "")
    
    db_session = get_db_session()
    classifier = IntentClassifier()
    
    # Classify the query (same approach as in potts.py)
    classification_result = classifier.classify(query)
    top_intent = classification_result["top_intent"]
    top_category = classification_result["top_category"]
    
    # RAGEngine for reasoning
    engine = RAGEngine()
    reasoning_response = engine.get_response(query, top_intent)
    
    # If an agent action is needed, pick the correct agent
    agent_response = None
    if top_intent == "Meal-Logging":
        agent = MealLoggingAgent()
        agent_response = agent.process_query(query, top_intent, top_category)
    elif top_intent == "Meal-Planning-Recipes":
        agent = MealPlanningAgent()
        user = db_session.query(UserProfile).filter_by(name=data.get("user_name")).first()
        agent_response = agent.process_query(query, top_intent, top_category, user)
    elif top_intent == "Educational-Content":
        agent = EducationalAgent()
        agent_response = agent.process_query(query, top_intent, top_category)
    elif top_intent == "Personalized-Health-Advice":
        agent = PersonalizedAdviceAgent()
        user = db_session.query(UserProfile).filter_by(name=data.get("user_name")).first()
        agent_response = agent.process_query(query, top_intent, top_category, user)

    # Merge final answer from RAGEngine reasoning and any agent action
    final_answer = agent_response["final_answer"] if agent_response else reasoning_response["final_answer"]
    reasoning = reasoning_response["reasoning"]

    return {
        "classification": classification_result,
        "reasoning": reasoning,
        "answer": final_answer
    }

@app.get("/health")
async def health_check():
    return {"status": "healthy"}


if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8001)