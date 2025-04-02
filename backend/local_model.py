import json
import os
import logging
import requests
from dotenv import load_dotenv

from potts import IntentClassifier
from retriever import Retriever
from tools import meal_planning, meal_logging

# Logging configuration
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

load_dotenv()

DEFAULT_MODEL = "dietbot"
AGENT_LOOP_LIMIT = 3
OLLAMA_API_URL = "http://localhost:11434/api/chat"

class LocalModel:
    def __init__(self) -> None:
        """Initialize model components"""
        try:
            self.intent_classifier = IntentClassifier()
            self.retriever = Retriever()
        except Exception as e:
            logger.error(f"Failed to initialize components (IntentClassifier/Retriever): {e}")
            raise
        
        self.model = os.getenv('OLLAMA_MODEL', DEFAULT_MODEL)
        self.agent_loop_limit = AGENT_LOOP_LIMIT
        
        # Base system prompt
        self.system_prompt = (
            "You are an AI assistant whose primary goal is to answer user questions effectively. "
            "When a user's question lacks sufficient information, use the `retrieve_context` tool to find relevant information. "
            "If retrieving additional context doesn't help, ask the user to clarify their question for more details. "
            "Avoid excessive looping to find answers if the information is unavailable; instead, be transparent and admit if you don't know. "
            "Always use the `retrieve_context` tool to verify the question's relevance to nutrition before answering. "
            "If the tool indicates the question is out of scope, inform the user immediately."
        )

    def _call_ollama(self, messages):
        """Make API call to local Ollama model and process streaming response"""
        try:
            payload = {
                "model": self.model,
                "messages": messages
            }
            response = requests.post(OLLAMA_API_URL, json=payload, stream=True)
            response.raise_for_status()
            
            # Process the streaming response
            full_content = ""
            for line in response.iter_lines():
                if line:
                    try:
                        chunk = json.loads(line.decode('utf-8'))
                        content_chunk = chunk.get("message", {}).get("content", "")
                        full_content += content_chunk
                        
                        # Check if this is the final message
                        if chunk.get("done", False):
                            break
                    except json.JSONDecodeError:
                        logger.warning(f"Failed to decode JSON from line: {line}")
            
            return {"message": {"content": full_content}}
        except Exception as e:
            logger.error(f"Error calling Ollama API: {e}")
            raise

    def _parse_tool_calls(self, response_content):
        """Parse tool calls from Ollama response"""
        tool_calls = []
        
        # Look for retrieve_context tool call pattern in the response
        if "```tool_code" in response_content:
            # Find all instances of tool code blocks
            tool_blocks = response_content.split("```tool_code")
            for block in tool_blocks[1:]:  # Skip the first element which is before the first tool_code
                if "```" in block:
                    tool_code = block.split("```")[0].strip()
                    
                    # Extract function call and arguments
                    if "retrieve_context" in tool_code:
                        # Parse argument string from patterns like: retrieve_context("query text")
                        query_start = tool_code.find("retrieve_context") + len("retrieve_context")
                        query_text = tool_code[query_start:].strip()
                        
                        # Extract the query inside parentheses and quotes
                        if "(" in query_text and ")" in query_text:
                            query_inside = query_text.split("(")[1].split(")")[0]
                            # Remove quotes if present
                            query = query_inside.strip('"\'')
                            
                            tool_calls.append({
                                "id": f"tool_call_{len(tool_calls) + 1}",
                                "function": {
                                    "name": "retrieve_context",
                                    "arguments": json.dumps({"query": query})
                                }
                            })
        
        return tool_calls

    def _format_final_response(self, response_content, top_intent, collected_contexts):
        """Extract and format the final response from Ollama"""
        # Remove tool_code blocks for final response
        final_content = response_content
        while "```tool_code" in final_content:
            pre_tool = final_content.split("```tool_code")[0]
            post_tool = final_content.split("```tool_code")[1]
            if "```" in post_tool:
                post_tool = post_tool.split("```", 1)[1]
                final_content = pre_tool + post_tool
            else:
                final_content = pre_tool
        
        # Check if the response contains a JSON object
        try:
            # First look for JSON object in markdown code block
            json_match = None
            if "```json" in final_content:
                json_blocks = final_content.split("```json")
                for block in json_blocks[1:]:
                    if "```" in block:
                        json_text = block.split("```")[0].strip()
                        try:
                            json_match = json.loads(json_text)
                            break
                        except:
                            continue
            
            # If no JSON found in code blocks, try to find it in the text
            if not json_match:
                json_start = final_content.find('{')
                json_end = final_content.rfind('}')
                
                if json_start != -1 and json_end != -1:
                    json_str = final_content[json_start:json_end+1]
                    json_match = json.loads(json_str)
            
            if json_match and isinstance(json_match, dict):
                if "reasoning" in json_match and "final_answer" in json_match:
                    return {
                        "reasoning": f"Identified intent: {top_intent}. " + json_match.get("reasoning", ""),
                        "final_answer": json_match.get("final_answer", ""),
                        "detected_intent": top_intent,
                        "context_used": collected_contexts[-1] if collected_contexts else "",
                        "raw_content": response_content
                    }
        except Exception as e:
            logger.warning(f"Error parsing JSON from response: {e}")
        
        # If no valid JSON found, use the full text as the answer
        # Clean up the response by removing tool call mentions
        clean_content = final_content.replace("I'll use the `retrieve_context` tool to get some relevant information.", "")
        clean_content = clean_content.strip()
        
        return {
            "reasoning": f"Identified intent: {top_intent}.",
            "final_answer": clean_content,
            "detected_intent": top_intent,
            "context_used": collected_contexts[-1] if collected_contexts else "",
            "raw_content": response_content
        }

    # Response generation method with classification RAG
    def get_response(self, query: str, user_context: dict = None) -> dict:
        """Generate a response based on the query and user context"""
        if not query or not query.strip():
            return {
                "reasoning": "No valid query provided.",
                "final_answer": "Please provide a valid query.",
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

            # Prepare messages for Ollama API
            if top_intent == "Educational-Content" or top_intent == "Personalized-Health-Advice":
                system_content = (
                    f"User's intent is {top_intent}. "
                    "You are a nutrition expert. If asked about non-nutrition topics, respond that it's out of scope. "
                    "If you need more information, use the retrieve_context tool by writing:\n"
                    "```tool_code\nretrieve_context(\"your query here\")\n```\n"
                    f"If the user's intent is classified as {top_intent}, "
                    "your primary goal is to answer questions effectively using the provided context."
                )
                
                messages = [
                    {"role": "system", "content": system_content},
                    {"role": "user", "content": query}
                ]
            
            # Meal logging tool
            elif top_intent == "Meal-Logging":
                return meal_logging(query, user_context)
            
            # Meal planning tool
            elif top_intent == "Meal-Planning-Recipes":
                prompt_with_context = (
                    meal_planning(user_context) + "\n\n"
                    "You are a nutrition expert. Always avoid allergies, respect dislikes, "
                    "prioritize likes, and follow the user's diet/goal."
                )
                
                messages = [
                    {"role": "system", "content": prompt_with_context},
                    {"role": "user", "content": query}
                ]

            # Response generation
            collected_contexts = []
            final_answer = None
            
            # Initial call to Ollama
            ollama_response = self._call_ollama(messages)
            response_content = ollama_response.get("message", {}).get("content", "")
            
            # Agent loop for context retrieval
            count = 0
            while count < self.agent_loop_limit:
                # Check for tool calls in the response
                tool_calls = self._parse_tool_calls(response_content)
                
                if not tool_calls:
                    # No tool calls, assume final response
                    break
                
                # Process tool calls and add context
                for tool_call in tool_calls:
                    arguments = json.loads(tool_call["function"]["arguments"])
                    context = self.retriever.retrieve(arguments.get("query", query))
                    collected_contexts.append(context)
                    
                    if context.startswith("OUT_OF_SCOPE:"):
                        final_answer = context.replace("OUT_OF_SCOPE:", "").strip()
                        break
                    
                    # Add tool response to messages
                    messages.append({
                        "role": "assistant", 
                        "content": f"I need to retrieve context for: {arguments.get('query', query)}"
                    })
                    messages.append({
                        "role": "user", 
                        "content": f"Context: {context}\n\nPlease use this context to answer the original question."
                    })
                
                if final_answer:
                    break
                
                # Make another call to Ollama with updated context
                ollama_response = self._call_ollama(messages)
                response_content = ollama_response.get("message", {}).get("content", "")
                count += 1

            # Process final response
            if final_answer:
                return {
                    "reasoning": f"Identified intent: {top_intent}. Question out of scope",
                    "final_answer": final_answer,
                    "detected_intent": top_intent,
                    "context_used": collected_contexts[-1] if collected_contexts else ""
                }
            
            # Format the final response
            return self._format_final_response(response_content, top_intent, collected_contexts)

        except Exception as e:
            logger.error(f"Error getting response: {e}")
            return {
                "reasoning": f"Error occurred: {str(e)}",
                "final_answer": f"Error: {str(e)}",
                "detected_intent": None,
                "context_used": ""
            }

if __name__ == "__main__":
    engine = LocalModel()
    response = engine.get_response("What is the importance of protein?")
    print(json.dumps(response, indent=2)) 