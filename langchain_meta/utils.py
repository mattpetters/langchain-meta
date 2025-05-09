"""
Meta-specific utility functions for better integration with LangChain and LangGraph.
"""

import json
import re
import logging
from typing import Dict, Any, Optional, List, Type, cast

from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables.base import RunnableSequence, Runnable
from langchain_core.tools import StructuredTool
from langchain_core.language_models import BaseChatModel
from langchain_core.runnables import RunnablePassthrough

# Set up logging
logger = logging.getLogger(__name__)

def meta_agent_factory(
    llm: BaseChatModel,
    tools: List[StructuredTool] = None,
    system_prompt_text: str = "",
    output_schema = None,  # Optional structured output schema
    disable_streaming: bool = False,
    additional_tags: List[str] = None,
) -> RunnableSequence:
    """Create a Meta-specific agent with structured output support.
    
    Args:
        llm: Base language model to use
        tools: Optional list of tools for the agent to use
        system_prompt_text: Optional system prompt to override the default
        output_schema: Optional Pydantic schema for structured output
        disable_streaming: Whether to disable streaming
        additional_tags: Optional list of additional tags
        
    Returns:
        A runnable chain that can be used for structured output
    """
    # Use structured output if schema is provided
    if output_schema is not None:
        logger.debug(f"Using structured output with schema: {output_schema}")
        
        try:
            # Use with_structured_output but add it to a chain properly
            bound_llm_wrapper = llm.with_structured_output(output_schema, include_raw=False)
            
            # Important: Wrap with RunnablePassthrough to ensure it's a proper Runnable
            bound_llm = RunnablePassthrough() | bound_llm_wrapper
            
            # Set streaming explicitly if needed
            if disable_streaming:
                bound_llm = bound_llm.bind(stream=False)
        except Exception as e:
            logger.error(f"Error setting up structured output: {e}")
            # Fall back to regular LLM
            bound_llm = llm
    else:
        # No schema provided, just use regular LLM
        bound_llm = llm
    
    # Add any additional tags
    if additional_tags:
        if isinstance(additional_tags, list):
            bound_llm = bound_llm.bind(tags=additional_tags)
    
    # Create a basic prompt
    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt_text),
        MessagesPlaceholder(variable_name="messages"),
    ])
    
    # Return the full chain
    return prompt | bound_llm


def extract_json_response(content):
    """
    Extract JSON from various response formats.
    
    Handles:
    - Direct JSON objects
    - JSON in code blocks with backticks
    - JSON-like patterns in text
    
    Args:
        content: The response content to parse
        
    Returns:
        Parsed JSON dict or original content if parsing failed
    """
    if not isinstance(content, str):
        return content
        
    # Try direct JSON parsing
    content = content.strip()
    try:
        return json.loads(content)
    except:
        pass
        
    # Try to extract JSON from code blocks
    if "```" in content:
        # Try JSON code blocks
        match = re.search(r'```(?:json)?\s*(.*?)\s*```', content, re.DOTALL)
        if match:
            try:
                return json.loads(match.group(1).strip())
            except:
                pass
    
    # Try to find JSON objects in text
    match = re.search(r'({[\s\S]*?})', content)
    if match:
        try:
            return json.loads(match.group(1))
        except:
            pass
            
    # If we get here, return the original content
    return content 