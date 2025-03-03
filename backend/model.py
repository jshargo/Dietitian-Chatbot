import json
import os
import logging
from dotenv import load_dotenv
from openai import OpenAI

from potts import IntentClassifier
from retriever import Retriever
from tools import meal_planning, meal_logging

# Logging configuration
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

load_dotenv()

DEFAULT_MODEL = "gpt-4o-mini"
AGENT_LOOP_LIMIT = 3

class Model:
    def __init__(self) -> None:
        """Initialize model components"""
        openai_api_key = os.getenv('OPENAI_API_KEY')
        
        if not openai_api_key:
            raise ValueError("Missing OPENAI_API_KEY environment variable. "
                             "Please set it before running.")
        try:
            self.client = OpenAI(api_key=openai_api_key)
        except Exception as e:
            logger.error(f"Failed to initialize OpenAI client: {e}")
            raise

        try:
            self.intent_classifier = IntentClassifier()
            self.retriever = Retriever()
        except Exception as e:
            logger.error(f"Failed to initialize components (IntentClassifier/Retriever): {e}")
            raise
        
        self.model = os.getenv('OPENAI_MODEL', DEFAULT_MODEL)
        self.agent_loop_limit = AGENT_LOOP_LIMIT
        
        # Tools
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
        # Base system prompt
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

    # Response generation method with classification RAG
    def get_response(self, query: str, user_context: dict = None) -> dict:
        """Generate a response based on the query and user context"""
        if not query or not query.strip():
            return {
                "reasoning": "No valid query provided.",
                "final_answer": (
                    "Please provide a valid query."
                ),
                "detected_intent": None,
                "context_used": ""
            }

        try:
            query_embedding = self.retriever.embed_query(query)
            intent_result = self.intent_classifier.classify_from_embedding(query_embedding)
            top_intent = intent_result['top_intent']
            initial_context = self.retriever.retrieve(query)
            
            # Unrelated question handling
            if "OUT_OF_SCOPE" in initial_context:
                return {
                    "reasoning": initial_context.replace("OUT_OF_SCOPE:", "Question out of scope:"),
                    "final_answer": (
                        "This question is outside my nutrition expertise. "
                        "Please ask about food, nutrients, or health-related topics."
                    ),
                    "detected_intent": top_intent,
                    "context_used": initial_context
                }

            # RAG on vector store 
            if top_intent == "Educational-Content" or top_intent == "Personalized-Health-Advice":
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
            
            # Meal logging tool
            elif top_intent == "Meal-Logging":
                return meal_logging(query, user_context)
            
            # Meal planning tool
            elif top_intent == "Meal-Planning-Recipes":
                prompt_with_context = (
                    meal_planning(user_context) + "\n\n"
                    "You are a nutrition expert. Always avoid allergies, respect dislikes, "
                    "prioritize likes, and follow the user's diet/goal.\n"
                )
                updated_system_message = {
                    "role": "system",
                    "content": prompt_with_context
                }
                intent_aware_messages = [updated_system_message, {"role": "user", "content": query}]

            # Response generation
            collected_contexts = []
            final_answer = None

            loop_response = self.client.chat.completions.create(
                model=self.model,
                messages=intent_aware_messages,
                tools=self.tools,
                response_format=self.context_reasoning
            )

            count = 0
            while (hasattr(loop_response.choices[0].message, "tool_calls")
                   and loop_response.choices[0].message.tool_calls
                   and count < self.agent_loop_limit):
                
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

            # Final output
            if final_answer:
                response_content = {
                    "reasoning": f"Identified intent: {top_intent}. Question out of scope",
                    "final_answer": final_answer
                }
            else:
                # Grab the final LLM output and parse
                raw_content = getattr(loop_response.choices[0].message, "content", "")
                if final_answer:
                    response_content = {
                        "reasoning": f"Identified intent: {top_intent}. Question out of scope",
                        "final_answer": final_answer
                    }
                else:
                    response_content = {}
                    if raw_content:
                        try:
                            response_content = json.loads(raw_content)
                        except Exception:
                            json_start = raw_content.find('{')
                            json_end = raw_content.rfind('}')
                            if json_start != -1 and json_end != -1:
                                try:
                                    response_content = json.loads(raw_content[json_start:json_end+1])
                                except Exception:
                                    response_content = {}
                    if not response_content.get("final_answer"):
                        response_content["final_answer"] = raw_content if raw_content else "No final answer generated."          

            # return raw_content too
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
                "detected_intent": None,
            }

if __name__ == "__main__":
    engine = Model()
    response = engine.get_response("What is the importance of protein?")
    print(json.dumps(response, indent=2))