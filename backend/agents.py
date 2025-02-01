
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

    def process_query(self, query: str, intent: str, category: str, user: UserProfile) -> dict:
        """Process query with intent and user context"""
        raise NotImplementedError

class MealLoggingAgent(BaseAgent):
    def process_query(self, query: str, intent: str, category: str, user: UserProfile) -> dict:
        # Extract meal information using LLM
        prompt = f"Extract the meal information from this query: '{query}'. Return a JSON format with fields: dish_name, calories, protein, fat, carbs, fiber, sodium. If any nutritional value is unknown, estimate based on typical values."
        meal_info = self.get_completion(prompt)
        
        # Store in database
        try:
            session = get_db_session()
            new_intake = DailyNutrientIntake(
                user_id=user.id,
                date=date.today(),
                **meal_info
            )
            session.add(new_intake)
            session.commit()
            return {
                "final_answer": f"Successfully logged your meal: {meal_info['dish_name']}",
                "reasoning": f"Logged meal using intent: {intent}, category: {category}"
            }
        except Exception as e:
            return {
                "final_answer": f"Error logging meal: {str(e)}",
                "reasoning": f"Error logging meal: {str(e)}"
            }

class MealPlanningAgent(BaseAgent):
    def process_query(self, query: str, intent: str, category: str, user: UserProfile) -> dict:
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
            return {
                "final_answer": f"Error fetching recipe: {str(e)}",
                "reasoning": f"Error fetching recipe: {str(e)}"
            }

class EducationalAgent(BaseAgent):
    def process_query(self, query: str, intent: str, category: str, user: UserProfile) -> dict:
        relevant_chunks = retrieve_knowledge(query)
        context = "\n".join(relevant_chunks)
        
        prompt = f"""As a {category} assistant ({intent}), answer this query: '{query}'
        Context: {context}
        Provide a clear, structured response:"""
        
        return {
            "final_answer": self.get_completion(prompt),
            "reasoning": f"Used educational context with {len(relevant_chunks)} relevant chunks"
        }

class PersonalizedAdviceAgent(BaseAgent):
    def process_query(self, query: str, intent: str, category: str, user: UserProfile) -> dict:
        relevant_chunks = retrieve_knowledge(query)
        
        prompt = f"""As a {category} assistant ({intent}) for {user.age}yo {user.sex}:
        Health Profile: {user.height}cm, {user.weight}kg, {user.activity_level}
        Context: {' '.join(relevant_chunks)}
        Query: {query}
        Provide personalized advice:"""
        
        return {
            "final_answer": self.get_completion(prompt), 
            "reasoning": f"Personalized advice using {len(relevant_chunks)} health factors"
        } 