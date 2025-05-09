import os
import sys
import json
import logging
from pydantic.v1 import BaseModel, Field
from langchain_core.messages import HumanMessage, SystemMessage

# Set up logging to see what's happening
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger("langchain_meta")

# Import our ChatMetaLlama class
from langchain_meta import ChatMetaLlama

# JSON Schema for routing
class RouteSchema(BaseModel):
    """Schema for the route decision."""
    next: str = Field(description="What action to take next. Can be 'EmailAgent', 'ScribeAgent', 'TimeKeeperAgent', 'GeneralAgent', or 'END'.")

# Helper to normalize responses
def parse_route_response(response):
    """Parse the route response, handling different field names."""
    # If it's already a RouteSchema, return it
    if isinstance(response, RouteSchema):
        return response.next
    
    # If it's a string, try to parse it as JSON
    if isinstance(response, str):
        # Remove code blocks if present
        if response.startswith("```") and response.endswith("```"):
            response = response.replace("```json", "").replace("```", "").strip()
        
        try:
            data = json.loads(response)
            # Check for field names
            if "next" in data:
                return data["next"]
            elif "agent" in data:
                return data["agent"]
            elif "route" in data:
                return data["route"]
            else:
                print(f"Warning: Unexpected JSON structure: {data}")
                return None
        except Exception as e:
            print(f"Error parsing response: {e}")
            return None
    
    return None

# API Configuration
API_KEY = "LLM|1071001774872506|g7XuFtow17z5d4_CaK8YauPy2YM"
BASE_URL = "https://api.llama.com/v1/"
MODEL_NAME = "Llama-4-Maverick-17B-128E-Instruct-FP8"

print(f"API Key (first 10 chars): {API_KEY[:10]}...")
print(f"Base URL: {BASE_URL}")
print(f"Using model: {MODEL_NAME}")

# Create the ChatMetaLlama instance
llm = ChatMetaLlama(
    api_key=API_KEY,
    base_url=BASE_URL,
    model=MODEL_NAME
)

# Test LangGraph compatibility with structured output for supervisor node
print("\n--- Testing LangGraph compatibility with structured output ---")
try:
    # Create the supervisor-style system prompt
    system_prompt = """You are a supervisor agent that decides which agent should handle a request.
    Your job is to analyze the conversation and decide which specialized agent should handle the current user query.
    
    Available agents are:
    - EmailAgent: Handles emails, messages, communication tasks
    - ScribeAgent: Handles note-taking, summarization, documentation
    - TimeKeeperAgent: Handles anything related to time, scheduling, dates
    - GeneralAgent: Handles general questions and tasks
    - END: End the conversation
    
    You must respond with just the name of the agent that should handle the request.
    """
    
    # Test with structured output approach
    structured_llm = llm.with_structured_output(RouteSchema)
    
    # Simple message
    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content="What time is it?")
    ]
    
    print("Sending message: 'What time is it?'")
    # Get the routing decision
    result = structured_llm.invoke(messages)
    
    print(f"Raw response: {result}")
    route_decision = parse_route_response(result)
    print(f"Normalized route: {route_decision}")
    
    # Test another message
    messages.append(HumanMessage(content="Can you help me draft an email to my boss?"))
    print("\nSending message: 'Can you help me draft an email to my boss?'")
    result2 = structured_llm.invoke(messages)
    
    print(f"Raw response: {result2}")
    route_decision2 = parse_route_response(result2)
    print(f"Normalized route: {route_decision2}")
    
    print("✅ LangGraph compatibility test successful!")
except Exception as e:
    print(f"❌ LangGraph compatibility test failed: {e}")
    import traceback
    traceback.print_exc()

print("\n--- All tests completed ---") 