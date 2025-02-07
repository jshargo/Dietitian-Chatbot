import os
import logging
import requests
from dotenv import load_dotenv

# Logging configuration
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

load_dotenv()

def safe_float(value):
    try:
        return float(value)
    except (ValueError, TypeError):
        return 0.0

def meal_logging(query: str, user_context: dict = None) -> dict:
    """Handle meal logging intent using the Nutritionix API."""
    dish_name = (query or "").strip()

    nutritionix_app_id = os.getenv("NUTRITIONIX_APP_ID")
    nutritionix_api_key = os.getenv("NUTRITIONIX_API_KEY")
    if not nutritionix_app_id or not nutritionix_api_key:
        return {
            "reasoning": "Meal logging requested, but missing Nutritionix credentials.",
            "final_answer": "Sorry, I'm missing the credentials to log your meal right now.",
            "detected_intent": "Meal-logging",
            "context_used": dish_name
        }
    
    try:
        response = requests.post(
            "https://trackapi.nutritionix.com/v2/natural/nutrients",
            headers={
                "x-app-id": nutritionix_app_id,
                "x-app-key": nutritionix_api_key,
                "Content-Type": "application/json"
            },
            json={"query": dish_name}
        )
        response.raise_for_status()
    except Exception as ex:
        logger.error(f"Nutritionix request failed: {ex}")
        return {
            "reasoning": "Error contacting Nutritionix API",
            "final_answer": "I'm sorry, I had trouble logging that meal. Please try again later.",
            "detected_intent": "Meal-logging",
            "context_used": dish_name
        }
    
    nutrition_data = response.json()
    if "foods" not in nutrition_data or len(nutrition_data["foods"]) == 0:
        return {
            "reasoning": "Meal logging identified. No nutritional data available.",
            "final_answer": "Couldn't retrieve nutritional information for that meal.",
            "detected_intent": "Meal-logging",
            "context_used": dish_name
        }
    
    food = nutrition_data["foods"][0]
    macros = {
        "calories": safe_float(food.get("nf_calories", 0)),
        "protein": safe_float(food.get("nf_protein", 0)),
        "fat": safe_float(food.get("nf_total_fat", 0)),
        "carbs": safe_float(food.get("nf_total_carbohydrate", 0)),
        "fiber": safe_float(food.get("nf_dietary_fiber", 0)),
        "sodium": safe_float(food.get("nf_sodium", 0))
    }
    
    macros_summary = (
        f"Meal: {dish_name}\n"
        f"Calories: {macros['calories']} kcal\n"
        f"Protein: {macros['protein']} g\n"
        f"Fat: {macros['fat']} g\n"
        f"Carbohydrates: {macros['carbs']} g\n"
        f"Fiber: {macros['fiber']} g\n"
        f"Sodium: {macros['sodium']} mg\n"
    )
    
    return {
        "reasoning": f"Meal logging identified. Processed dish: {dish_name}",
        "final_answer": f"Your meal has been logged successfully.\n{macros_summary}",
        "detected_intent": "Meal-logging",
        "context_used": dish_name
    }

def meal_planning(user_context: dict) -> str:
    """Build a meal planning prompt based on user context."""
    if not user_context:
        return (
            "User's intent is Meal-Planning-Recipes. "
            "No user context provided. Suggest a general meal."
        )
    
    allergies = ', '.join(user_context.get("allergies", [])) or "None"
    likes = ', '.join(user_context.get("likes", [])) or "No specific preferences"
    dislikes = ', '.join(user_context.get("dislikes", [])) or "None"
    diet = user_context.get("diet", "No Restriction")
    goal = user_context.get("goal", "Maintain Weight")
    
    prompt = (
        f"You are a nutritionist creating meal plans. Critical constraints:\n"
        f"- ABSOLUTELY NEVER suggest {allergies} (life-threatening allergies)\n"
        f"- Avoid: {dislikes}\n"
        f"- Prioritize: {likes}\n"
        f"- Diet: {diet}\n"
        f"- Goal: {goal}\n\n"
        "Thinking process:\n"
        "1. Check allergies first - remove prohibited ingredients\n"
        "2. Eliminate disliked foods\n"
        "3. Select preferred ingredients\n"
        f"4. Calculate nutritional needs based on {goal}\n"
        "5. Propose a meal meeting all criteria"
    )
    
    return prompt