import os
import sys
import logging
from pydantic.v1 import BaseModel, Field
from langchain_core.messages import HumanMessage, SystemMessage

# Set up logging to see what's happening
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger("langchain_meta")

# Import our ChatMetaLlama class
from langchain_meta import ChatMetaLlama

# API Configuration with real credentials
API_KEY = os.environ.get("META_API_KEY","TEST_KEY")
BASE_URL = os.environ.get("META_NATIVE_API_BASE_URL", "https://api.llama.com/v1/")
MODEL_NAME = os.environ.get("META_MODEL_NAME", "Llama-4-Maverick-17B-128E-Instruct-FP8")

print(f"API Key (first 10 chars): {API_KEY[:10]}...")
print(f"Base URL: {BASE_URL}")
print(f"Using model: {MODEL_NAME}")

# Create the ChatMetaLlama instance
llm = ChatMetaLlama(
    api_key=API_KEY,
    base_url=BASE_URL,
    model=MODEL_NAME
)

# Test 1: Simple message without tools
print("\n--- Test 1: Simple message without tools ---")
try:
    response = llm.invoke("Hello Llama! Can you give me a quick intro?")
    print(f"Response: {response.content}")
    print("✅ Test 1 successful!")
except Exception as e:
    print(f"❌ Test 1 failed: {e}")
    import traceback
    traceback.print_exc()

# Test 2: Using the with_structured_output method
print("\n--- Test 2: Using with_structured_output ---")
try:
    # Define a simple schema
    class WeatherInfo(BaseModel):
        """Information about the weather in a location."""
        location: str = Field(description="The city and state, e.g. San Francisco, CA")
        temperature: int = Field(description="The temperature in Fahrenheit")
        condition: str = Field(description="The weather condition (sunny, cloudy, rain, etc.)")
    
    # Get a structured response
    structured_llm = llm.with_structured_output(WeatherInfo)
    
    # Invoke with a question that should produce structured data
    result = structured_llm.invoke("What's the weather like in Miami today?")
    
    print(f"Structured Response: {result}")
    
    # The response might be a string containing JSON, let's try to parse it
    import json
    if result is None:
        raise ValueError("Got None response from structured output")
    elif isinstance(result, str):
        # Try to parse JSON string
        weather_data = json.loads(result)
        # Check if we need to extract from a nested structure
        if "location" in weather_data:
            print(f"Location: {weather_data['location']}")
            
            # Handle both temperature as a number directly or as an object
            if isinstance(weather_data.get('temperature'), dict) and 'value' in weather_data['temperature']:
                temp_value = weather_data['temperature']['value']
            else:
                temp_value = weather_data.get('temperature')
            print(f"Temperature: {temp_value}°F")
            
            # Handle variations in condition field name (condition vs conditions)
            condition = weather_data.get('condition') or weather_data.get('conditions')
            print(f"Condition: {condition}")
        elif "weather" in weather_data:
            print(f"Location: {weather_data['weather']['location']}")
            print(f"Temperature: {weather_data['weather']['temperature']['value']}°F")
            print(f"Condition: {weather_data['weather']['condition']}")
        else:
            print(f"Unexpected JSON structure: {weather_data}")
    elif hasattr(result, "location") and hasattr(result, "temperature") and hasattr(result, "condition"):
        # Direct WeatherInfo object
        print(f"Location: {result.location}")
        print(f"Temperature: {result.temperature}°F")
        print(f"Condition: {result.condition}")
    else:
        print(f"Unexpected response type: {type(result)}")
        print(f"Response content: {result}")
    
    print("✅ Test 2 successful!")
except Exception as e:
    print(f"❌ Test 2 failed: {e}")
    import traceback
    traceback.print_exc()

# Test 3: Using bind_tools with a simple tool
print("\n--- Test 3: Using bind_tools with a tool ---")
try:
    # Define a simple tool schema
    class CalculatorTool(BaseModel):
        """A simple calculator tool that adds two numbers."""
        a: int = Field(description="First number")
        b: int = Field(description="Second number")
    
    # Bind the tool
    tool_llm = llm.bind_tools([CalculatorTool])
    
    # Create messages that will encourage tool use
    messages = [
        SystemMessage(content="You are a helpful assistant with access to a calculator. Use it when asked about math."),
        HumanMessage(content="What is 123 + 456? Please use your calculator tool.")
    ]
    
    # Invoke
    result = tool_llm.invoke(messages)
    
    print(f"Response with tool: {result.content}")
    if hasattr(result, "tool_calls") and result.tool_calls:
        print(f"Tool calls: {result.tool_calls}")
    print("✅ Test 3 successful!")
except Exception as e:
    print(f"❌ Test 3 failed: {e}")
    import traceback
    traceback.print_exc()

print("\n--- All tests completed ---") 