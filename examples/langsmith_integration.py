"""
Example demonstrating LangSmith integration with ChatMetaLlama.

This example shows how to use LangSmith to trace and debug your Meta Llama
model interactions, including tool usage and token tracking.

Prerequisites:
- LangSmith account (https://smith.langchain.com/)
- Meta API key (https://www.llama.com/products/llama-api/)

Setup:
1. Set your environment variables:
   - LANGSMITH_API_KEY: Your LangSmith API key
   - LANGSMITH_PROJECT: Your project name (will be created if it doesn't exist)
   - META_API_KEY: Your Meta API key

2. Run this script:
   python langsmith_integration.py

3. Visit your LangSmith dashboard to view the traces
"""

import os
import sys
from datetime import datetime

# Set LangSmith environment variables
# You can set these in your environment instead of here
os.environ["LANGSMITH_TRACING"] = "true"
os.environ["LANGSMITH_PROJECT"] = "Meta-LLM-Examples"

# Check for required API keys
if "META_API_KEY" not in os.environ:
    print("Error: META_API_KEY environment variable not set.")
    print("Get an API key from https://www.llama.com/products/llama-api/")
    sys.exit(1)

if "LANGSMITH_API_KEY" not in os.environ:
    print("Error: LANGSMITH_API_KEY environment variable not set.")
    print("Get a LangSmith API key from https://smith.langchain.com/")
    sys.exit(1)

try:
    import llama_api_client
except ImportError:
    print("Error: llama_api_client package not installed.")
    print("Install it with: pip install llama-api-client")
    sys.exit(1)

from langchain_core.callbacks import get_usage_metadata_callback
from langchain_core.messages import HumanMessage
from langchain_core.tools import tool

from langchain_meta import ChatMetaLlama

# Function to print separator
def print_section(title):
    """Print a section title with separators."""
    print("\n" + "="*50)
    print(f" {title} ".center(50, "-"))
    print("="*50 + "\n")

# Initialize Meta Llama client and LangChain wrapper
client = llama_api_client.LlamaAPIClient(
    api_key=os.environ.get("META_API_KEY"),
    base_url=os.environ.get("META_API_BASE_URL", "https://api.llama.com/v1/")
)

llm = ChatMetaLlama(
    client=client,
    model_name="Llama-4-Maverick-17B-128E-Instruct-FP8",
    temperature=0.5,
    verbose=True
)

# Create some tools
@tool
def get_weather(location: str) -> str:
    """Get the current weather in a given location."""
    return f"The weather in {location} is 72°F and sunny."

@tool
def get_time(timezone: str = "UTC") -> str:
    """Get the current time in a given timezone."""
    return f"The current time in {timezone} is {datetime.now()}."

# Example 1: Basic LLM completion with token tracking
print_section("Basic LLM completion")
print("Sending a simple query to Meta LLM...\n")

# Use the callback to track token usage
with get_usage_metadata_callback() as cb:
    response = llm.invoke("What are the key features of Meta's Llama 4 models?")
    
    print(f"Response: {response.content}\n")
    
    # Display token usage
    print("Token Usage:")
    for model, usage in cb.usage_metadata.items():
        print(f"  Model: {model}")
        print(f"  Input tokens: {usage.get('input_tokens', 0)}")
        print(f"  Output tokens: {usage.get('output_tokens', 0)}")
        print(f"  Total tokens: {usage.get('total_tokens', 0)}")

# Example 2: Tool usage with token tracking
print_section("Tool usage")
print("Using Meta LLM with tool calling...\n")

# Bind tools to the model
llm_with_tools = llm.bind_tools([get_weather, get_time])

with get_usage_metadata_callback() as cb:
    # Call the model with a question that should trigger tool use
    response = llm_with_tools.invoke("What's the weather in San Francisco?")
    
    # Print the response and any tool calls made
    print("Response:", response.content)
    
    if response.tool_calls:
        print("\nTool Calls:")
        for i, tool_call in enumerate(response.tool_calls):
            print(f"  Tool {i+1}: {tool_call['function']['name']}")
            print(f"  Arguments: {tool_call['function']['arguments']}")
    
    # Display token usage
    print("\nToken Usage:")
    for model, usage in cb.usage_metadata.items():
        print(f"  Model: {model}")
        print(f"  Input tokens: {usage.get('input_tokens', 0)}")
        print(f"  Output tokens: {usage.get('output_tokens', 0)}")
        print(f"  Total tokens: {usage.get('total_tokens', 0)}")

# Example 3: Chat history with token tracking
print_section("Chat history")
print("Creating a chat history with Meta LLM...\n")

# Create a conversation history
messages = [
    HumanMessage(content="Hello, I'd like to know about Meta Llama models."),
]

with get_usage_metadata_callback() as cb:
    # First message
    response = llm.invoke(messages)
    print("Human: Hello, I'd like to know about Meta Llama models.")
    print(f"AI: {response.content}\n")
    
    # Add AI response to history
    messages.append(response)
    
    # Second message
    messages.append(HumanMessage(content="What is the largest model available?"))
    response = llm.invoke(messages)
    print("Human: What is the largest model available?")
    print(f"AI: {response.content}\n")
    
    # Display token usage
    print("Token Usage:")
    for model, usage in cb.usage_metadata.items():
        print(f"  Model: {model}")
        print(f"  Input tokens: {usage.get('input_tokens', 0)}")
        print(f"  Output tokens: {usage.get('output_tokens', 0)}")
        print(f"  Total tokens: {usage.get('total_tokens', 0)}")

print_section("Summary")
print("All interactions have been traced to LangSmith.")
print("Visit your dashboard to view details: https://smith.langchain.com/")
print("Project name:", os.environ.get("LANGSMITH_PROJECT", "default"))
print("\nToken usage has been tracked and is available in each trace.")
print("You can use this data to estimate costs and optimize your application.") 