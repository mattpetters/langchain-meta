#!/usr/bin/env python3
"""
Test script for the fixed Meta structured output implementation,
particularly focusing on RunnableSequence integration.
"""

import os
import sys
from typing import Dict, Any, List
from pydantic import BaseModel, Field

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.prompts import PromptTemplate

# Import our custom Meta implementation
from langchain_meta import ChatMetaLlama

# Simple test schema
class RouteSchema(BaseModel):
    """Schema for routing between agents."""
    next: str = Field(description="The agent to route to next")

def test_direct_structured_output():
    """Test the fixed structured output implementation directly."""
    print("\n=== Testing Direct Structured Output ===")
    
    # Initialize the LLM
    api_key = os.environ.get("META_API_KEY")
    if not api_key:
        print("ERROR: META_API_KEY environment variable not set")
        return False
    
    # Create our LLM
    llm = ChatMetaLlama(
        api_key=api_key,
        base_url=os.environ.get("META_API_BASE_URL", "https://api.llama.com/v1/"),
        model_name=os.environ.get("META_MODEL_NAME", "Llama-4-Maverick-17B-128E-Instruct-FP8"),
        temperature=0.1  # Low temperature for predictable outputs
    )
    
    # Create a structured output version
    llm_with_schema = llm.with_structured_output(RouteSchema)
    
    # Test with a simple routing request
    messages = [
        SystemMessage(content="You are a routing agent that decides which specialized agent should handle a request."),
        HumanMessage(content="I need help with my calendar and scheduling.")
    ]
    
    print("Calling with_structured_output directly...")
    result = llm_with_schema.invoke(messages)
    print(f"Result type: {type(result)}")
    print(f"Result: {result}")
    
    return True

def test_simple_invocation():
    """Test a simpler form of structured output invocation."""
    print("\n=== Testing Simple Invocation ===")
    
    # Initialize the LLM
    api_key = os.environ.get("META_API_KEY")
    if not api_key:
        print("ERROR: META_API_KEY environment variable not set")
        return False
    
    # Create our LLM
    llm = ChatMetaLlama(
        api_key=api_key,
        base_url=os.environ.get("META_API_BASE_URL", "https://api.llama.com/v1/"),
        model_name=os.environ.get("META_MODEL_NAME", "Llama-4-Maverick-17B-128E-Instruct-FP8"),
        temperature=0.1
    )
    
    # Create a structured output version
    llm_with_schema = llm.with_structured_output(RouteSchema)
    
    # Test with a simple input string
    prompt = "As a routing agent, determine where to send a request about 'email issues'."
    
    print("Calling with simple string input...")
    result = llm_with_schema.invoke(prompt)
    print(f"Result type: {type(result)}")
    print(f"Result: {result}")
    
    return True

def test_standard_template():
    """Test with a standard template and direct handling."""
    print("\n=== Testing with Standard Template ===")
    
    # Initialize the LLM
    api_key = os.environ.get("META_API_KEY")
    if not api_key:
        print("ERROR: META_API_KEY environment variable not set")
        return False
    
    # Create our LLM
    llm = ChatMetaLlama(
        api_key=api_key,
        base_url=os.environ.get("META_API_BASE_URL", "https://api.llama.com/v1/"),
        model_name=os.environ.get("META_MODEL_NAME", "Llama-4-Maverick-17B-128E-Instruct-FP8"),
        temperature=0.1
    )
    
    # Create a structured output version
    llm_with_schema = llm.with_structured_output(RouteSchema)
    
    # Create a simple template
    template = """You are a routing agent that decides which specialized agent should handle a request.
    
    User request: {input}
    
    Decide which agent should handle this request based on its content."""
    
    prompt = PromptTemplate.from_template(template)
    
    print("Calling with formatted template...")
    # Format the prompt manually
    formatted_prompt = prompt.format(input="I need help with my email.")
    
    # Send the formatted prompt to the LLM
    result = llm_with_schema.invoke(formatted_prompt)
    print(f"Result type: {type(result)}")
    print(f"Result: {result}")
    
    return True

if __name__ == "__main__":
    print("Testing fixed structured output implementation...")
    
    success = True
    
    # Test each component
    try:
        if not test_direct_structured_output():
            success = False
            print("Direct structured output test failed!")
    except Exception as e:
        success = False
        print(f"Error in direct test: {e}")
    
    try:
        if not test_simple_invocation():
            success = False
            print("Simple invocation test failed!")
    except Exception as e:
        success = False
        print(f"Error in simple invocation test: {e}")
    
    try:
        if not test_standard_template():
            success = False
            print("Standard template test failed!")
    except Exception as e:
        success = False
        print(f"Error in standard template test: {e}")
    
    print("\n=== Test Summary ===")
    if success:
        print("✅ All tests completed successfully!")
        sys.exit(0)
    else:
        print("❌ Some tests failed. See output for details.")
        sys.exit(1) 