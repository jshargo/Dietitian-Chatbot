import numpy as np
import json
import os
import logging
import requests
from dotenv import load_dotenv
from openai import OpenAI
from embeddings import cosine_similarity, embed_texts
from potts import IntentClassifier
from sentence_transformers import SentenceTransformer
import pandas as pd

#logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

load_dotenv()

# Model config
DEFAULT_MODEL = "gpt-4o-mini"
AGENT_LOOP_LIMIT = 1

class Model:
    
    def __init__(self) -> None:
        """Initialize model"""
        try:
            self.client = OpenAI(
                api_key=os.getenv('OPENAI_API_KEY')
            )
            # Initialize intent classifier
            self.intent_classifier = IntentClassifier()
            
            # Shared embedding model
            self.embed_model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
            
            # Load knowledge base
            self.knowledge_df = pd.read_csv("../data/embeddings.csv")
            
            # Convert string representations to numpy arrays
            self.knowledge_embeddings = np.array(
                self.knowledge_df['embedding']
                .apply(lambda x: np.array(eval(x), dtype=np.float32))  # Safe conversion
                .tolist()
            )
            self.knowledge_texts = self.knowledge_df['sentence_chunk'].tolist()
        except Exception as e:
            logger.error(f"Failed to initialize components: {e}")
            raise
        
        # Configuration
        self.model = os.getenv('OPENAI_MODEL', DEFAULT_MODEL)
        self.agent_loop_limit = AGENT_LOOP_LIMIT
        
        # Tool definitions
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
        
        # Response schema
        self.context_reasoning = {
            "type": "json_schema",
            "json_schema": {
                "name": "context_reasoning",
                "schema": {
                "type": "object",
                "properties": {
                    "reasoning": { "type": "string" },
                    "final_answer": { "type": "string" }
                },
                "required": ["reasoning", "final_answer"],
                "additionalProperties": False
                },
                "strict": True
            }
        }
        
        # Initial system message
        self.messages = [
           {
                "role": "system",
                "content": (
                    "You are an AI assistant whose primary goal is to answer user questions effectively. "
                    "When a user's question lacks sufficient information, use the `retrieve_context` tool to find relevant information. "
                    "If retrieving additional context doesn't help, ask the user to clarify their question for more details. "
                    "Avoid excessive looping to find answers if the information is unavailable; instead, be transparent and admit if you don't know."
                    "Always use the `retrieve_context` tool to verify the question's relevance to nutrition before answering. If the tool indicates the question is out of scope, inform the user immediately."
                )
            }
        ]

    def embed_query(self, query: str) -> np.ndarray:
        return self.embed_model.encode([query])[0]

    def retrieve_context(self, query: str) -> str:
        try:
            query_embedding = self.embed_query(query)
            
            # Normalize embeddings for cosine similarity
            query_norm = query_embedding / np.linalg.norm(query_embedding)
            knowledge_norms = np.linalg.norm(self.knowledge_embeddings, axis=1)
            
            # Compute cosine similarities correctly
            similarities = np.dot(self.knowledge_embeddings, query_norm) / knowledge_norms
            max_score = np.max(similarities)
            
            if max_score < 0.30:
                return f"OUT_OF_SCOPE: This question is outside my nutrition expertise. Please ask about food, nutrients, or health-related topics. (score: {max_score:.2f})"
                
            most_relevant_idx = np.argmax(similarities)
            return f"Knowledge Source (score: {max_score:.2f}): {self.knowledge_texts[most_relevant_idx]}"
            
        except Exception as e:
            logger.error(f"Retrieval error: {str(e)}")
            return "Error accessing knowledge base"

    def handle_meal_logging(self, query: str) -> dict:
        # Use the entire query as the dish name for simplicity
        dish_name = query.strip()
        
        # Prepare the Nutritionix API request
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
        
    def get_response(self, query: str) -> dict:
        query_embedding = self.embed_query(query)
        intent_result = self.intent_classifier.classify_from_embedding(query_embedding)
        top_intent = intent_result['top_intent']
        
        # Intercept meal logging requests
        if top_intent.lower() == "meal-logging":
            return self.handle_meal_logging(query)
        
        initial_context = self.retrieve_context(query)
        if "OUT_OF_SCOPE" in initial_context:
            return {
                "reasoning": initial_context.replace("OUT_OF_SCOPE:", "Question out of scope:"),
                "final_answer": "This question is outside my nutrition expertise. Please ask about food, nutrients, or health-related topics.",
                "detected_intent": top_intent,
                "context_used": initial_context
            }
        
        updated_system_message = {
            "role": "system",
            "content": (
                f"User's intent is {top_intent}. "
                "You are a nutrition expert. "
                "If asked about non-nutrition topics, respond that it's out of scope. "
                "Always use retrieve_context tool before answering."
                f" If the user's intent is classified as {top_intent}, "
                "your primary goal is to answer questions effectively using the provided context. "
                "When needed, use tools to retrieve additional information and explain your reasoning."
            )
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
            while loop_response.choices[0].message.tool_calls and count < self.agent_loop_limit:
                tool_call_results_message = []
                for tool_call in loop_response.choices[0].message.tool_calls:
                    arguments = json.loads(tool_call.function.arguments)
                    context = self.retrieve_context(arguments.get("query", query))
                    collected_contexts.append(context)
                    
                    if context.startswith("OUT_OF_SCOPE:"):
                        final_answer = context.replace("OUT_OF_SCOPE:", "").strip()
                        break

                    tool_call_results_message.append({
                        "role": "tool",
                        "content": context,
                        "tool_call_id": tool_call.id
                    })
                
                if final_answer:
                    break

                intent_aware_messages.extend([
                    loop_response.choices[0].message,
                    *tool_call_results_message
                ])

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
                response_content = json.loads(loop_response.choices[0].message.content) if loop_response.choices[0].message.content else {
                    "reasoning": "Failed to generate proper response",
                    "final_answer": "I'm having trouble answering that. Please try rephrasing your question."
                }

            return {
                "reasoning": f"Identified intent: {top_intent}. " + response_content["reasoning"],
                "final_answer": response_content["final_answer"],
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
