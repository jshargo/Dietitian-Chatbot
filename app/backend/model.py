from typing import List, Dict, Any, Tuple
import numpy as np
import json
import os
import logging
from dotenv import load_dotenv
from openai import OpenAI
from pydantic import BaseModel, Field
from sentence_transformers import SentenceTransformer
import pandas as pd

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

# Constants
DEFAULT_MODEL = "gpt-4o-mini"
AGENT_LOOP_LIMIT = 3

class RAGEngine:
    """
    Retrieval-Augmented Generation (RAG) engine that combines OpenAI's language models
    with a local knowledge base for context-aware responses.
    """
    
    def __init__(self) -> None:
        """Initialize the RAG engine with OpenAI client and configuration."""
        try:
            self.client = OpenAI(
                api_key=os.getenv('OPENAI_API_KEY')
            )
        except Exception as e:
            logger.error(f"Failed to initialize OpenAI client: {e}")
            raise
        
        # Configuration
        self.model = os.getenv('OPENAI_MODEL', DEFAULT_MODEL)
        self.agent_loop_limit = AGENT_LOOP_LIMIT
        
        # Replace sample data with CSV embeddings
        self.data, self.embeddings = self._load_embeddings("../data/embeddings.csv")
        
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
        
        # Initialize local embedding model (same as potts.py)
        self.embedder = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
        
        # Initial system message
        self.messages = [
           {
                "role": "system",
                "content": (
                    "You are an AI assistant that first analyzes the user's intent category, then provides "
                    "expert answers using relevant context. Follow these steps:\n"
                    "1. Identify which intent category the question belongs to\n"
                    "2. Retrieve relevant information using the tool\n"
                    "3. Synthesize a clear answer using the retrieved context\n"
                    "4. If unsure, ask clarifying questions"
                )
            }
        ]

    def _load_embeddings(self, csv_path: str) -> Tuple[List[str], List[np.ndarray]]:
        """Load precomputed embeddings from CSV"""
        if not os.path.exists(csv_path):
            raise FileNotFoundError(f"Embeddings file not found: {csv_path}")
        
        df = pd.read_csv(csv_path)
        
        # Convert string embeddings to numpy arrays
        df['embedding'] = df['embedding'].apply(
            lambda x: np.array(eval(x)) if isinstance(x, str) else x
        )
        
        return df['sentence_chunk'].tolist(), df['embedding'].tolist()

    def retrieve_context(self, query: str) -> str:
        """
        Retrieve the most relevant context for the given query using embedding similarity.
        
        Args:
            query (str): The user's query to find relevant context for
            
        Returns:
            str: The most relevant context from the knowledge base
        """
        try:
            # Encode query using local model
            query_embedding = self.embedder.encode([query])[0]
            
            # Calculate similarities with precomputed embeddings
            similarities = [
                cosine_similarity(query_embedding, emb)
                for emb in self.embeddings
            ]
            
            most_relevant_idx = np.argmax(similarities)
            return self.data[most_relevant_idx]
        except Exception as e:
            logger.error(f"Error retrieving context: {e}")
            raise

    def get_response(self, query: str, intent_classification: str) -> Dict[str, Any]:
        """
        Modified to include intent classification in the reasoning process
        """
        # Add intent context to the message history
        self.messages.append({
            "role": "system",
            "content": f"Current intent classification: {intent_classification}"
        })
        self.messages.append({"role": "user", "content": query})
        try:
            intermediate_steps = []
            # Initial function call to retrieve context
            initial_response = self.client.chat.completions.create(
                model=self.model,
                messages=self.messages,
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
                    # Add tool input step
                    intermediate_steps.append({
                        "explanation": "Tool Input",
                        "output": f"Function: {tool_call.function.name}, Arguments: {json.dumps(arguments)}"
                    })
                    
                    context = self.retrieve_context(arguments.get("query", query))
                    tool_call_results_message.append({
                        "role": "tool",
                        "content": context,
                        "tool_call_id": tool_call.id
                    })
                    
                    # Add tool response step
                    intermediate_steps.append({
                        "explanation": "Tool Response",
                        "output": context
                    })
                
                # Update messages with context and reasoning instruction
                self.messages.extend([
                    loop_response.choices[0].message,
                    *tool_call_results_message
                ])

                # Final call for tool response and reasoning
                loop_response = self.client.chat.completions.create(
                    model=self.model,
                    messages=self.messages,
                    tools=self.tools,
                    response_format=self.context_reasoning
                )
                count += 1
            
            final_response = json.loads(loop_response.choices[0].message.content) if loop_response.choices[0].message.content is not None else {"reasoning": "Stuck in loop", "final_answer": "Error: Stuck in loop"}
            # Append assistant response
            self.messages.append({"role": "assistant", "content": final_response["final_answer"]})
            return {
                "intermediate_steps": intermediate_steps if intermediate_steps else [],
                "reasoning": final_response["reasoning"],
                "final_answer": final_response["final_answer"]
            }
            
        except Exception as e:
            logger.error(f"Error getting response: {e}")
            return {
                "intermediate_steps": [{
                    "explanation": "Error",
                    "output": str(e)
                }],
                "reasoning": f"Error occurred: {str(e)}",
                "final_answer": f"Error: {str(e)}"
            }

if __name__ == "__main__":
    engine = RAGEngine()
    response = engine.get_response("What is machine learning?", "Information Retrieval")
    print(json.dumps(response, indent=2))
