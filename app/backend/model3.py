from typing import Dict, Any
import numpy as np
import json
import os
import logging
from dotenv import load_dotenv
from openai import OpenAI
from embeddings import cosine_similarity, embed_texts
from potts import IntentClassifier

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
        except Exception as e:
            logger.error(f"Failed to initialize components: {e}")
            raise
        
        # Configuration
        self.model = os.getenv('OPENAI_MODEL', DEFAULT_MODEL)
        self.agent_loop_limit = AGENT_LOOP_LIMIT
        
        # Sample data - in production this would come from a database
        self.data = [
            "Python is a versatile programming language used for web development, data analysis, and more.",
            "OpenAI provides advanced AI models like GPT-4 that support function calling.",
            "Function calling allows external tools to be integrated seamlessly into chatbots.",
            "Machine learning is a subset of artificial intelligence that focuses on building algorithms.",
            "The Turing test is a benchmark for evaluating an AI's ability to mimic human intelligence.",
            "Transformers are a type of neural network architecture that powers modern AI systems.",
            "Kotlin is a modern programming language, widely used for Android app development.",
            "Docker and Kubernetes are essential tools for containerized application deployment.",
        ]
        
        # Tool definitions
        self.tools = [
            {
                "type": "function",
                "function": {
                    "name": "retrieve_context",
                    "description": "Retrieve relevant context from the dataset based on the query.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "query": {
                                "type": "string",
                                "description": "The user's query to find relevant context."
                            }
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
                )
            }
        ]

    def retrieve_context(self, query: str) -> str:
        """
        Retrieve the most relevant context for the given query using embedding similarity.
        
        Args:
            query (str): The user's query to find relevant context for
            
        Returns:
            str: The most relevant context from the knowledge base
        """
        try:
            data_embeddings = embed_texts(self.data)
            query_embedding = embed_texts([query])[0].reshape(1, -1)
            
            similarities = [
                cosine_similarity(query_embedding, data_embedding.reshape(1, -1)) 
                for data_embedding in data_embeddings
            ]
            
            most_relevant_idx = np.argmax(similarities)
            return self.data[most_relevant_idx]
        except Exception as e:
            logger.error(f"Error retrieving context: {e}")
            raise

    def get_response(self, query: str) -> Dict[str, Any]:
        # Classify user intent first
        intent_result = self.intent_classifier.classify(query)
        top_intent = intent_result['top_intent']
        
        # Update system message with intent classification
        updated_system_message = {
            "role": "system",
            "content": (
                f"If the user's intent is classified as {top_intent}. " +
                "Your primary goal is to answer questions effectively using this context. " +
                "When needed, use tools to retrieve additional information. " +
                "Always explain your reasoning including the intent classification."
            )
        }
        
        # Create fresh message chain with intent context
        intent_aware_messages = [updated_system_message, {"role": "user", "content": query}]
        
        try:
            # Use intent-aware messages instead of stored messages
            initial_response = self.client.chat.completions.create(
                model=self.model,
                messages=intent_aware_messages,
                tools=self.tools,
                response_format=self.context_reasoning
            )
            
            loop_response = initial_response
            count = 0
            while loop_response.choices[0].message.tool_calls and count < self.agent_loop_limit:
                # Execute all tool calls
                tool_call_results_message = []
                for tool_call in loop_response.choices[0].message.tool_calls:
                    arguments = json.loads(tool_call.function.arguments)
                    
                    context = self.retrieve_context(arguments.get("query", query))
                    tool_call_results_message.append({
                        "role": "tool",
                        "content": context,
                        "tool_call_id": tool_call.id
                    })
                
                
                # Update messages with context and reasoning instruction
                intent_aware_messages.extend([
                    loop_response.choices[0].message,
                    *tool_call_results_message
                ])

                # Final call for tool response and reasoning
                loop_response = self.client.chat.completions.create(
                    model=self.model,
                    messages=intent_aware_messages,
                    tools=self.tools,
                    response_format=self.context_reasoning
                )
                count += 1
            
            final_response = json.loads(loop_response.choices[0].message.content) if loop_response.choices[0].message.content is not None else {"reasoning": "Stuck in loop", "final_answer": "Error: Stuck in loop"}
            # Append assistant response
            intent_aware_messages.append({"role": "assistant", "content": final_response["final_answer"]})
            
            # Update final response with intent information
            return {
                "reasoning": f"Identified intent: {top_intent}. " + final_response["reasoning"],
                "final_answer": final_response["final_answer"],
                "detected_intent": top_intent
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
                "detected_intent": top_intent
            }

if __name__ == "__main__":
    engine = Model()
    response = engine.get_response("What is the importance of protein?")
    print(json.dumps(response, indent=2))
