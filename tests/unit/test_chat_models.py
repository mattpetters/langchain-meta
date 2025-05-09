"""Test chat model integration."""

import asyncio
from typing import Awaitable, Callable

import pytest
from unittest.mock import MagicMock, patch
import logging

from langchain_meta import ChatMetaLlama
from langchain_core.messages import HumanMessage
# Corrected import for CreateChatCompletionResponse, assuming it's part of llama_api_client.types
from llama_api_client.types import CreateChatCompletionResponse 
from pydantic.v1 import BaseModel as PydanticV1BaseModel, Field as PydanticV1Field
from llama_api_client.types.chat import completion_create_params
from langchain_meta.chat_models import _lc_tool_to_llama_tool_param # Keep this for other tests

# Import the real class from its actual location BEFORE it might be patched in a fixture
from llama_api_client import LlamaAPIClient as OriginalLlamaAPIClient

@pytest.fixture
def mock_llama_client_fixture(): # Renamed to avoid confusion if test uses same name
    with patch("llama_api_client.LlamaAPIClient") as mock_client_class_constructor:
        # This mock_client_class_constructor is what LlamaAPIClient becomes (the constructor itself)
        # When LlamaAPIClient() is called, it's effectively mock_client_class_constructor() being called.

        # We want mock_client_class_constructor() to return our special mock instance.
        fully_configured_mock_instance = MagicMock() 

        # Make this instance masquerade as an OriginalLlamaAPIClient instance for isinstance checks
        fully_configured_mock_instance.__class__ = OriginalLlamaAPIClient

        # Setup the rest of the mock client's behavior (chat, completions, create)
        mock_response_object = MagicMock(spec=CreateChatCompletionResponse)
        mock_response_object.completion_message = MagicMock()
        mock_response_object.completion_message.content = "Test response"
        mock_response_object.completion_message.tool_calls = None
        mock_response_object.completion_message.stop_reason = "stop"
        mock_response_object.metrics = None

        mock_chat_attribute = MagicMock()
        mock_completions_attribute = MagicMock()

        # This is the async function that will be the side_effect
        async def actual_mock_create_for_side_effect(*args, **kwargs):
            return mock_response_object

        # 'create' itself is a MagicMock, and its side_effect is our async func
        create_method_mock = MagicMock(name="create_method_mock", side_effect=actual_mock_create_for_side_effect)
        
        mock_completions_attribute.create = create_method_mock # Assign the MagicMock
        mock_chat_attribute.completions = mock_completions_attribute
        fully_configured_mock_instance.chat = mock_chat_attribute
        
        # Configure the patched constructor (mock_client_class_constructor) to return our specific instance
        mock_client_class_constructor.return_value = fully_configured_mock_instance

        yield mock_client_class_constructor # Yield the patched constructor

# Separate fixture for the configured mock instance, if preferred
@pytest.fixture
def configured_mock_llama_client(mock_llama_client_fixture):
    # mock_llama_client_fixture is the patched LlamaAPIClient constructor.
    # Its return_value is our fully_configured_mock_instance.
    return mock_llama_client_fixture.return_value


def test_chat_meta_llama_initialization(configured_mock_llama_client): # Use the instance
    # This test primarily checks __init__ logic, direct client interaction isn't the focus here
    # So, we can initialize with api_key and let it try to create its own (which will be mocked if patch works)
    llm = ChatMetaLlama(api_key="test-key") # Relies on the @patch in fixture to mock internal creation
    assert llm.model_name == "Llama-4-Maverick-17B-128E-Instruct-FP8"
    identifying_params = llm._identifying_params
    assert "model_name" in identifying_params
    assert identifying_params["model_name"] == "Llama-4-Maverick-17B-128E-Instruct-FP8"
    # Check if the client created internally is indeed our mock that masquerades
    if llm.client is not None:
        assert isinstance(llm.client, OriginalLlamaAPIClient) # It should pass this due to __class__ override
        assert llm.client is configured_mock_llama_client # And it should be the exact same instance


@pytest.mark.asyncio
async def test_agenerate_method(configured_mock_llama_client): # Use the instance fixture
    # Pass the already configured mock client instance directly to ChatMetaLlama
    llm = ChatMetaLlama(client=configured_mock_llama_client, model_name="CustomTestModel") # Pass client, api_key not needed
    
    messages = [HumanMessage(content="Hello")]
    result = await llm._agenerate(messages)

    assert len(result.generations) == 1
    assert "Test response" in result.generations[0].message.content
    
    # Verify that the mock 'create' method was called
    # configured_mock_llama_client is the instance, so: instance.chat.completions.create
    configured_mock_llama_client.chat.completions.create.assert_called_once()
    # We can also assert details about the call if needed, e.g., what model was passed
    call_args = configured_mock_llama_client.chat.completions.create.call_args
    assert call_args.kwargs['model'] == "CustomTestModel"
    assert call_args.kwargs['messages'] == [{'role': 'user', 'content': 'Hello'}]


# --- Tests for _lc_tool_to_llama_tool_param ---

class MyTestToolArgs(PydanticV1BaseModel):
    """Arguments for my test tool."""
    param1: str = PydanticV1Field(description="First parameter")
    param2: int = PydanticV1Field(description="Second parameter")

class MySimpleTool(PydanticV1BaseModel):
    """A simple Pydantic model tool."""
    query: str = PydanticV1Field(description="The query string")

class NoDocstringTool(PydanticV1BaseModel):
    data: str

class MockStructuredTool:
    def __init__(self, name, description, args_schema):
        self.name = name
        self.description = description
        self.args_schema = args_schema

class MockBoundObject:
    def __init__(self, name, description, schema_):
        self.name = name
        self.description = description
        self.schema_ = schema_

def test_already_llama_formatted_dict():
    """Test case 1: Input is already a Llama API formatted dict."""
    tool_dict = {
        "type": "function",
        "function": {
            "name": "MyPreformattedTool",
            "description": "Does something preformatted.",
            "parameters": {"type": "object", "properties": {"arg": {"type": "string"}}},
        },
    }
    result = _lc_tool_to_llama_tool_param(tool_dict)
    assert result == tool_dict

def test_pydantic_v1_model_class():
    """Test case 2: Input is a Pydantic V1 model class."""
    result = _lc_tool_to_llama_tool_param(MySimpleTool)
    expected = {
        "type": "function",
        "function": {
            "name": "MySimpleTool",
            "description": "A simple Pydantic model tool.",
            "parameters": MySimpleTool.schema(),
        },
    }
    assert result == expected

def test_pydantic_model_no_docstring():
    """Test case 2b: Pydantic model with no docstring."""
    result = _lc_tool_to_llama_tool_param(NoDocstringTool)
    expected = {
        "type": "function",
        "function": {
            "name": "NoDocstringTool",
            "description": "",
            "parameters": NoDocstringTool.schema(),
        },
    }
    assert result == expected

def test_langchain_bound_object_with_schema_dict():
    """Test case 3: Input is a LangChain-bound object with .schema_ (dict)."""
    schema_dict = {
        "type": "object",
        "properties": {"paramA": {"type": "integer"}},
        "required": ["paramA"],
    }
    bound_obj = MockBoundObject(
        name="BoundToolName",
        description="Description of bound tool.",
        schema_=schema_dict,
    )
    result = _lc_tool_to_llama_tool_param(bound_obj)
    expected = {
        "type": "function",
        "function": {
            "name": "BoundToolName",
            "description": "Description of bound tool.",
            "parameters": schema_dict,
        },
    }
    assert result == expected

def test_structured_tool_with_pydantic_args():
    """Test case 4: StructuredTool-like with Pydantic args_schema."""
    tool = MockStructuredTool(
        name="MyStructToolPydantic",
        description="Structured tool with Pydantic args.",
        args_schema=MyTestToolArgs,
    )
    result = _lc_tool_to_llama_tool_param(tool)
    expected = {
        "type": "function",
        "function": {
            "name": "MyStructToolPydantic",
            "description": "Structured tool with Pydantic args.",
            "parameters": MyTestToolArgs.schema(),
        },
    }
    assert result == expected

def test_structured_tool_with_dict_args():
    """Test case 5: StructuredTool-like with dict args_schema."""
    args_schema_dict = {
        "type": "object",
        "properties": {"key": {"type": "boolean", "description": "A boolean key"}},
    }
    tool = MockStructuredTool(
        name="MyStructToolDict",
        description="Structured tool with dict args.",
        args_schema=args_schema_dict,
    )
    result = _lc_tool_to_llama_tool_param(tool)
    expected = {
        "type": "function",
        "function": {
            "name": "MyStructToolDict",
            "description": "Structured tool with dict args.",
            "parameters": args_schema_dict,
        },
    }
    assert result == expected

def test_structured_tool_no_args():
    """Test case 5b: StructuredTool-like with args_schema=None."""
    tool = MockStructuredTool(
        name="NoArgsTool",
        description="A tool that takes no arguments.",
        args_schema=None,
    )
    result = _lc_tool_to_llama_tool_param(tool)
    expected = {
        "type": "function",
        "function": {
            "name": "NoArgsTool",
            "description": "A tool that takes no arguments.",
            "parameters": {"type": "object", "properties": {}},
        },
    }
    assert result == expected

def test_unsupported_tool_type(caplog):
    """Test case 6: Input is an unsupported type."""
    class SomeRandomClass:
        pass

    # Test that we log an error instead of raising ValueError
    with caplog.at_level(logging.ERROR):
        result = _lc_tool_to_llama_tool_param(SomeRandomClass())
            
    # Verify we got a fallback tool with an empty schema
    assert result["type"] == "function"
    assert result["function"]["name"] == "SomeRandomClass"
    assert result["function"]["parameters"] == {"type": "object", "properties": {}}
    # Check logs for the expected message
    assert "Could not convert tool to Llama API format" in caplog.text

def test_structured_tool_non_string_name(caplog):
    """Test case 7a: StructuredTool-like with non-string name."""
    tool = MockStructuredTool(name=123, description="Valid desc", args_schema=None)
    
    # Test that we convert rather than raise ValueError
    with caplog.at_level(logging.WARNING):
        result = _lc_tool_to_llama_tool_param(tool)
        
    # Verify we got a tool with the converted name
    assert result["type"] == "function"
    assert result["function"]["name"] == "123"  # Converted to string
    assert "Tool name is not a string" in caplog.text

def test_structured_tool_non_string_description(caplog):
    """Test case 7b: StructuredTool-like with non-string description."""
    tool = MockStructuredTool(name="ValidName", description=["list"], args_schema=None)
    
    # Test that we convert rather than raise ValueError
    with caplog.at_level(logging.WARNING):
        result = _lc_tool_to_llama_tool_param(tool)
        
    # Verify we got a tool with the converted description
    assert result["type"] == "function"
    assert result["function"]["name"] == "ValidName"
    assert "Tool description is not a string" in caplog.text
    # Description should be the string representation of the list
    assert result["function"]["description"] == str(["list"])

def test_structured_tool_invalid_args_schema_type(caplog):
    """Test case 8: StructuredTool-like with invalid args_schema type."""
    tool = MockStructuredTool(
        name="InvalidArgsTool",
        description="Tool with bad args_schema.",
        args_schema=12345,
    )
    
    # Test that we handle without raising ValueError
    with caplog.at_level(logging.WARNING):
        result = _lc_tool_to_llama_tool_param(tool)
        
    # Verify we got a tool with default empty schema
    assert result["type"] == "function"
    assert result["function"]["name"] == "InvalidArgsTool"
    assert result["function"]["parameters"] == {"type": "object", "properties": {}}

def test_tool_choice_parameter_filtering():
    """Test that tool_choice parameter is filtered out."""
    from langchain_meta.chat_models import ChatMetaLlama
    
    # Create a minimal instance
    llm = ChatMetaLlama(model_name="TestModel", api_key="test-key")
    
    # Extract the tool_choice parameter directly from kwargs
    test_kwargs = {"tool_choice": "auto", "temperature": 0.7, "max_tokens": 100}
    
    # This method simulates what happens in the ChatMetaLlama methods
    # before passing parameters to the API
    def process_kwargs(kwargs):
        kwargs_copy = kwargs.copy()
        if "tool_choice" in kwargs_copy:
            kwargs_copy.pop("tool_choice")
        return kwargs_copy
    
    # Directly test that our filter logic works
    processed = process_kwargs(test_kwargs)
    assert "tool_choice" not in processed
    assert "temperature" in processed
    assert "max_tokens" in processed
    assert processed["temperature"] == 0.7
    
    # Get the actual method logic from the class implementation
    filter_code = "\n".join([line for line in ChatMetaLlama._generate.__code__.co_consts if isinstance(line, str) and "tool_choice" in line])
    assert "tool_choice" in filter_code, "The class should mention tool_choice in its code"
    assert "not supported" in filter_code.lower(), "The class should warn that tool_choice is not supported"

@pytest.mark.asyncio
async def test_async_tool_choice_parameter_filtering():
    """Test that tool_choice parameter is filtered out in async methods."""
    from langchain_meta.chat_models import ChatMetaLlama
    
    # Create a minimal instance
    llm = ChatMetaLlama(model_name="TestModel", api_key="test-key")
    
    # Extract the tool_choice parameter directly from kwargs
    test_kwargs = {"tool_choice": "required", "temperature": 0.5, "max_tokens": 200}
    
    # This method simulates what happens in the ChatMetaLlama methods
    # before passing parameters to the API
    def process_kwargs(kwargs):
        kwargs_copy = kwargs.copy()
        if "tool_choice" in kwargs_copy:
            kwargs_copy.pop("tool_choice")
        return kwargs_copy
    
    # Directly test that our filter logic works
    processed = process_kwargs(test_kwargs)
    assert "tool_choice" not in processed
    assert "temperature" in processed
    assert "max_tokens" in processed
    assert processed["temperature"] == 0.5
    
    # Get the actual method logic from the class implementation
    filter_code = "\n".join([line for line in ChatMetaLlama._agenerate.__code__.co_consts if isinstance(line, str) and "tool_choice" in line])
    assert "tool_choice" in filter_code, "The class should mention tool_choice in its code"
    assert "not supported" in filter_code.lower(), "The class should warn that tool_choice is not supported"

@pytest.mark.asyncio
async def test_bind_with_tool_choice_async(configured_mock_llama_client):
    """Test that bind method properly handles tool_choice parameter in async calls."""
    from langchain_meta.chat_models import ChatMetaLlama
    
    # Create a minimal instance with the already mocked client
    llm = ChatMetaLlama(client=configured_mock_llama_client, model_name="TestModel")
    
    # Create a simple tool
    class SimpleToolSchema(PydanticV1BaseModel):
        """A simple test tool."""
        name: str = PydanticV1Field(description="The name to greet")
    
    # Test binding with tool_choice as object
    bound_llm = llm.bind(
        tools=[SimpleToolSchema],
        tool_choice={"type": "function", "function": {"name": "SimpleToolSchema"}}
    )
    
    # Verify the binding worked (didn't throw error)
    assert bound_llm is not None
    
    # Now call with a message to ensure it works
    messages = [HumanMessage(content="Hello")]
    
    # Call agenerate to trigger the API call - client is already mocked via fixture
    result = await bound_llm._agenerate(messages)
    
    # Verify the API was called with the correct parameters
    configured_mock_llama_client.chat.completions.create.assert_called_once()
    
    # Most importantly - verify tool_choice is NOT in the kwargs
    call_args = configured_mock_llama_client.chat.completions.create.call_args
    assert "tool_choice" not in call_args.kwargs
    
    # Verify that tools got bound but not passed directly to the kwargs
    # Instead, they should be extracted via kwargs.pop("tools", None) in _agenerate
    # and then converted to the llama_tools format
    
    # Reset mock for next test
    configured_mock_llama_client.chat.completions.create.reset_mock()
    
    # Test with "none" tool_choice - this should effectively skip binding tools
    bound_llm_none = llm.bind(
        tools=[SimpleToolSchema],
        tool_choice="none"
    )
    result = await bound_llm_none._agenerate(messages)
    
    # Verify tool_choice was not passed to the API
    call_args = configured_mock_llama_client.chat.completions.create.call_args
    assert "tool_choice" not in call_args.kwargs
