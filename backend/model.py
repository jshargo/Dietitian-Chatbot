import json
import os
import logging
import requests
from dotenv import load_dotenv
from openai import OpenAI

from potts import IntentClassifier
from retriever import Retriever
from context import build_meal_planning_prompt

# Logging configuration
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

load_dotenv()

# Model configuration
DEFAULT_MODEL = "gpt-4o-mini"
AGENT_LOOP_LIMIT = 1

class Model:
    def __init__(self) -> None:
        """Initialize model components."""
        try:
            self.client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))
            # Initialize intent classifier
            self.intent_classifier = IntentClassifier()
            # Instantiate the Retriever (which loads the knowledge base and embedding model)
            self.retriever = Retriever()
        except Exception as e:
            logger.error(f"Failed to initialize components: {e}")
            raise
        
        # Other configurations
        self.model = os.getenv('OPENAI_MODEL', DEFAULT_MODEL)
        self.agent_loop_limit = AGENT_LOOP_LIMIT
        
        # Tool definitions for chat completions
        self.tools = [
            {
                "type": "function",
                "function": {
                    "name": "retrieve_context",
                    "description": "Retrieve relevant context from the nutrition knowledge base",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "query": {"type": "string", "description": "The user's query to find relevant context."}
                        },
                        "required": ["query"]
                    }
                }
            }
        ]
        
        # Response schema for tool calls
        self.context_reasoning = {
            "type": "json_schema",
            "json_schema": {
                "name": "context_reasoning",
                "schema": {
                    "type": "object",
                    "properties": {
                        "reasoning": {"type": "string"},
                        "final_answer": {"type": "string"}
                    },
                    "required": ["reasoning", "final_answer"],
                    "additionalProperties": False
                },
                "strict": True
            }
        }
        
        # Initial system prompt message
        self.messages = [
            {
                "role": "system",
                "content": (
                    "You are an AI assistant whose primary goal is to answer user questions effectively. "
                    "When a user's question lacks sufficient information, use the `retrieve_context` tool to find relevant information. "
                    "If retrieving additional context doesn't help, ask the user to clarify their question for more details. "
                    "Avoid excessive looping to find answers if the information is unavailable; instead, be transparent and admit if you don't know. "
                    "Always use the `retrieve_context` tool to verify the question's relevance to nutrition before answering. "
                    "If the tool indicates the question is out of scope, inform the user immediately."
                )
            }
        ]
    
    def handle_meal_logging(self, query: str, user_context: dict = None) -> dict:
        """Handle meal logging intent using the Nutritionix API."""
        dish_name = query.strip()
        
        nutritionix_app_id = os.getenv("NUTRITIONIX_APP_ID")
        nutritionix_api_key = os.getenv("NUTRITIONIX_API_KEY")
        payload = {"query": dish_name}
        headers = {
            "x-app-id": nutritionix_app_id,
            "x-app-key": nutritionix_api_key,
            "Content-Type": "application/json"
        }
        
        response = requests.post(
            "https://trackapi.nutritionix.com/v2/natural/nutrients",
            headers=headers,
            json=payload
        )
    
        nutrition_data = response.json()
        if "foods" not in nutrition_data or len(nutrition_data["foods"]) == 0:
            return {
                "reasoning": "Meal logging identified. No nutritional data available.",
                "final_answer": "Couldn't retrieve nutritional information for that meal.",
                "detected_intent": "Meal-logging",
                "context_used": dish_name
            }
        
        food = nutrition_data["foods"][0]
        
        def safe_float(value):
            try:
                return float(value)
            except (ValueError, TypeError):
                return 0.0
        
        calories = safe_float(food.get("nf_calories", 0))
        protein  = safe_float(food.get("nf_protein", 0))
        fat      = safe_float(food.get("nf_total_fat", 0))
        carbs    = safe_float(food.get("nf_total_carbohydrate", 0))
        fiber    = safe_float(food.get("nf_dietary_fiber", 0))
        sodium   = safe_float(food.get("nf_sodium", 0))
        
        macros_summary = (
            f"Meal: {dish_name}\n"
            f"Calories: {calories} kcal\n"
            f"Protein: {protein} g\n"
            f"Fat: {fat} g\n"
            f"Carbohydrates: {carbs} g\n"
            f"Fiber: {fiber} g\n"
            f"Sodium: {sodium} mg\n"
        )
        
        confirmation_message = (
            f"Your meal has been logged successfully.\n{macros_summary}"
        )
        
        return {
            "reasoning": f"Meal logging identified. Processed dish: {dish_name}",
            "final_answer": confirmation_message,
            "detected_intent": "Meal-logging",
            "context_used": dish_name
        }
        
    def get_response(self, query: str, user_context: dict = None) -> dict:
        """Generate a response based on the query and user context."""
        query_embedding = self.retriever.embed_query(query)
        intent_result = self.intent_classifier.classify_from_embedding(query_embedding)
        top_intent = intent_result['top_intent']
        initial_context = self.retriever.retrieve(query)
        if "OUT_OF_SCOPE" in initial_context:
            return {
                "reasoning": initial_context.replace("OUT_OF_SCOPE:", "Question out of scope:"),
                "final_answer": "This question is outside my nutrition expertise. Please ask about food, nutrients, or health-related topics.",
                "detected_intent": top_intent,
                "context_used": initial_context
            }
        
        if top_intent == "Educational-Content":
            updated_system_message = {
                "role": "system",
                "content": (
                    f"User's intent is {top_intent}. "
                    "You are a nutrition expert. If asked about non-nutrition topics, respond that it's out of scope. "
                    "Always use retrieve_context tool before answering. "
                    f"If the user's intent is classified as {top_intent}, "
                    "your primary goal is to answer questions effectively using the provided context. "
                    "When needed, use tools to retrieve additional information and explain your reasoning."
                )
            }
            intent_aware_messages = [updated_system_message, {"role": "user", "content": query}]
        else:
            if top_intent == "Meal-Logging":
                return self.handle_meal_logging(query, user_context)
            
            if top_intent == "Meal-Planning-Recipes":
                prompt_with_context = (
                    build_meal_planning_prompt(user_context)
                    + "\n\n"
                    "You are a nutrition expert. Always avoid allergies, respect dislikes, prioritize likes, and follow the user's diet/goal.\n"
                )
            else:
                prompt_with_context = (
                    f"User's intent is {top_intent}. "
                    "You are a nutrition expert. If asked about non-nutrition topics, respond with out-of-scope."
                )
            
            updated_system_message = {
                "role": "system",
                "content": prompt_with_context
            }
            intent_aware_messages = [updated_system_message, {"role": "user", "content": query}]
        
        try:
            collected_contexts = []
            final_answer = None
            
            loop_response = self.client.chat.completions.create(
                model=self.model,
                messages=intent_aware_messages,
                tools=self.tools,
                response_format=self.context_reasoning
            )
            
            count = 0
            while (hasattr(loop_response.choices[0].message, "tool_calls") and 
                   loop_response.choices[0].message.tool_calls and 
                   count < self.agent_loop_limit):
                message_obj = loop_response.choices[0].message
                intent_aware_messages.append({
                    "role": "assistant",
                    "content": message_obj.content,
                    "tool_calls": [
                        {
                            "id": tool_call.id,
                            "type": "function",
                            "function": {
                                "name": tool_call.function.name,
                                "arguments": tool_call.function.arguments
                            }
                        } for tool_call in message_obj.tool_calls
                    ]
                })
                
                for tool_call in message_obj.tool_calls:
                    arguments = json.loads(tool_call.function.arguments)
                    context = self.retriever.retrieve(arguments.get("query", query))
                    collected_contexts.append(context)
                    
                    if context.startswith("OUT_OF_SCOPE:"):
                        final_answer = context.replace("OUT_OF_SCOPE:", "").strip()
                        break

                    intent_aware_messages.append({
                        "role": "tool",
                        "tool_call_id": tool_call.id,
                        "name": tool_call.function.name,
                        "content": context
                    })
                
                if final_answer:
                    break
                
                loop_response = self.client.chat.completions.create(
                    model=self.model,
                    messages=intent_aware_messages,
                    tools=self.tools,
                    response_format=self.context_reasoning
                )
                count += 1
            
            if final_answer:
                response_content = {
                    "reasoning": f"Identified intent: {top_intent}. Question out of scope",
                    "final_answer": final_answer
                }
            else:
                raw_content = getattr(loop_response.choices[0].message, "content", "")
                response_content = {}
                if raw_content:
                    try:
                        response_content = json.loads(raw_content)
                    except Exception:
                        # Fallback: try extracting the JSON substring if extra formatting is present
                        json_start = raw_content.find('{')
                        json_end = raw_content.rfind('}')
                        if json_start != -1 and json_end != -1:
                            try:
                                response_content = json.loads(raw_content[json_start:json_end+1])
                            except Exception:
                                response_content = {}
                if not response_content:
                    response_content = {
                        "reasoning": "Failed to generate proper response",
                        "final_answer": "I'm having trouble answering that. Please try rephrasing your question."
                    }

            return {
                "reasoning": f"Identified intent: {top_intent}. " + response_content.get("reasoning", ""),
                "final_answer": response_content.get("final_answer", ""),
                "detected_intent": top_intent,
                "context_used": collected_contexts[-1] if collected_contexts else ""
            }
            
        except Exception as e:
            logger.error(f"Error getting response: {e}")
            return {
                "intermediate_steps": [{
                    "explanation": "Error",
                    "output": str(e)
                }],
                "reasoning": f"Error occurred: {str(e)}",
                "final_answer": f"Error: {str(e)}",
                "detected_intent": top_intent,
            }

if __name__ == "__main__":
    engine = Model()
    response = engine.get_response("What is the importance of protein?")
    print(json.dumps(response, indent=2))