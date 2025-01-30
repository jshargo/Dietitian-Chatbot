from model import LocalLLM
from database import get_db_session, UserProfile, DailyNutrientIntake
from rag import retrieve_knowledge
from datetime import date
import requests
import os
from typing import Dict, List

RAPIDAPI_KEY = os.getenv("RAPIDAPI_KEY")

class BaseAgent:
    def __init__(self):
        self.llm = LocalLLM()
    
    def get_completion(self, prompt: str) -> str:
        try:
            return self.llm.generate(prompt)
        except Exception as e:
            return f"Error generating response: {str(e)}"

class MealLoggingAgent(BaseAgent):
    def process_query(self, query: str, user_id: int) -> str:
        # Extract meal information using LLM
        prompt = f"Extract the meal information from this query: '{query}'. Return a JSON format with fields: dish_name, calories, protein, fat, carbs, fiber, sodium. If any nutritional value is unknown, estimate based on typical values."
        meal_info = self.get_completion(prompt)
        
        # Store in database
        try:
            session = get_db_session()
            new_intake = DailyNutrientIntake(
                user_id=user_id,
                date=date.today(),
                **meal_info
            )
            session.add(new_intake)
            session.commit()
            return f"Successfully logged your meal: {meal_info['dish_name']}"
        except Exception as e:
            return f"Error logging meal: {str(e)}"

class MealPlanningAgent(BaseAgent):
    def process_query(self, query: str) -> str:
        # Use LLM to extract recipe requirements
        prompt = f"Extract the key recipe requirements from: '{query}'. Focus on main ingredients and dish type."
        recipe_requirements = self.get_completion(prompt)
        
        # Call Spoonacular API
        headers = {
            'X-RapidAPI-Key': RAPIDAPI_KEY,
            'X-RapidAPI-Host': 'spoonacular-recipe-food-nutrition-v1.p.rapidapi.com'
        }
        
        try:
            response = requests.get(
                'https://spoonacular-recipe-food-nutrition-v1.p.rapidapi.com/recipes/search',
                headers=headers,
                params={'query': recipe_requirements}
            )
            recipe_data = response.json()
            return self._format_recipe_response(recipe_data)
        except Exception as e:
            return f"Error fetching recipe: {str(e)}"

class EducationalAgent(BaseAgent):
    def process_query(self, query: str) -> str:
        # Get relevant chunks from vector store
        relevant_chunks = retrieve_knowledge(query)
        
        # Use RAG with LocalLLM
        context = "\n".join(relevant_chunks)
        prompt = f"""Based on the following context, answer the question: '{query}'
        
        Context:
        {context}
        
        Answer:"""
        
        return self.get_completion(prompt)

class PersonalizedAdviceAgent(BaseAgent):
    def process_query(self, query: str, user_id: int) -> str:
        # Get user profile
        session = get_db_session()
        user = session.query(UserProfile).filter_by(id=user_id).first()
        
        # Get relevant educational content
        relevant_chunks = retrieve_knowledge(query)
        
        # Combine user profile with educational content
        prompt = f"""Given a person with these characteristics:
        - Age: {user.age}
        - Sex: {user.sex}
        - Height: {user.height}cm
        - Weight: {user.weight}kg
        - Activity Level: {user.activity_level}
        
        And this nutritional context:
        {' '.join(relevant_chunks)}
        
        Provide personalized advice for their question: '{query}'"""
        
        return self.get_completion(prompt) 