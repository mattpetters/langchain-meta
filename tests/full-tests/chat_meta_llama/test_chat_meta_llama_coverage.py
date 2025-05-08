"""Tests to increase code coverage for ChatMetaLlama class."""

import asyncio
import json
import os
from unittest.mock import MagicMock, patch, AsyncMock
import sys

import pytest
from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    HumanMessage,
)
from langchain_core.tools import BaseTool
from llama_api_client import LlamaAPIClient
from llama_api_client.types.create_chat_completion_response import CreateChatCompletionResponse

from integration.chat_meta_llama import (
    ChatMetaLlama,
    _lc_message_to_llama_message_param,
    _lc_tool_to_llama_tool_param,
)


@pytest.fixture
def mock_llama_client():
    """Create a mock Llama API client for testing."""
    mock_client = MagicMock(spec=LlamaAPIClient)
    
    # Set up nested mocks for completions
    completions = MagicMock()
    chat = MagicMock()
    chat.completions = completions
    mock_client.chat = chat
    
    # Return a mocked completion response by default
    mock_response = MagicMock()
    mock_response.completion_message = MagicMock()
    mock_response.completion_message.content = "Test response"
    mock_response.completion_message.tool_calls = []
    mock_response.completion_message.stop_reason = "stop"
    mock_response.metrics = [MagicMock(metric="tokens", value=10)]
    
    mock_client.chat.completions.create.return_value = mock_response
    
    return mock_client


# Test URL defaults when unspecified
def test_client_init_default_url():
    """Test that the default URL is used when no base_url is provided."""
    with patch.dict(os.environ, {}, clear=True):
        with patch('llama_api_client.LlamaAPIClient') as mock_actual_client_cls:
            mock_instance = MagicMock()
            mock_actual_client_cls.return_value = mock_instance
            
            # Instantiate the model. _ensure_client_initialized might run here if llama_api_key is given.
            model = ChatMetaLlama(
                model_name="test-model",
                llama_api_key="test-key" 
            )
            
            # Reset the client and mock call count to test _ensure_client_initialized directly
            model.client = None
            mock_actual_client_cls.reset_mock() # Reset call count from initial __init__ call
            
            model._ensure_client_initialized()
            
            mock_actual_client_cls.assert_called_once()
            args, kwargs = mock_actual_client_cls.call_args
            assert kwargs['api_key'] == "test-key"
            assert kwargs['base_url'] == "https://api.llama.com/v1/"


# Test import error handling correctly
def test_client_init_import_error():
    """Test handling of ImportError when llama-api-client is not available."""
    with patch.dict(os.environ, {"META_API_KEY": "test-key"}, clear=True):
        model = ChatMetaLlama(model_name="test-model")
        model.client = None
        
        # To simulate ImportError for 'from llama_api_client import LlamaAPIClient',
        # we need to make 'llama_api_client' itself unimportable or LlamaAPIClient within it.
        # Patching sys.modules is a common way to simulate module-level import issues.
        with patch.dict(sys.modules, {'llama_api_client': None}):
            with pytest.raises(ImportError, match="llama-api-client package not found"):
                model._ensure_client_initialized()


# Test general error during client initialization
def test_client_init_other_error():
    """Test handling of other errors during client initialization."""
    with patch.dict(os.environ, {"META_API_KEY": "test-key"}, clear=True):
        model = ChatMetaLlama(model_name="test-model")
        model.client = None

        # Patch the LlamaAPIClient where it's imported from to raise an error during instantiation
        with patch('llama_api_client.LlamaAPIClient', side_effect=ValueError("Client init failed")):
            with pytest.raises(ValueError, match="Failed to initialize LlamaAPIClient: Client init failed"):
                model._ensure_client_initialized()


# Test handling of missing API key
def test_missing_api_key():
    """Test error when API key is not provided."""
    # Clear environment variables
    with patch.dict(os.environ, {}, clear=True):
        # Create model without API key
        model = ChatMetaLlama(model_name="test-model")
        
        # Force the client initialization check which should raise the error
        # First patch any existing attributes to ensure clean test
        if hasattr(model, 'client'):
            delattr(model, 'client')
        if hasattr(model, 'llama_api_key'):
            model.llama_api_key = None
            
        # Now the initialization should fail with key error
        with pytest.raises(ValueError, match="Meta Llama API key must be provided"):
            model._ensure_client_initialized()


# Test handling of list content in messages
def test_message_with_content_list():
    """Test handling of message content that is a list."""
    message = HumanMessage(content=["Hello", "World"])
    
    # Should raise ValueError since HumanMessage content must be a string
    with pytest.raises(ValueError, match="HumanMessage content for Llama API must be a string"):
        result = _lc_message_to_llama_message_param(message)


# Test validation for tools with missing attributes
def test_tool_missing_name():
    """Test error when tool is missing required attributes."""
    bad_tool = MagicMock(spec=BaseTool)
    delattr(bad_tool, 'name')
    
    with pytest.raises(ValueError, match="LangChain tool must have name, description, and args_schema"):
        _lc_tool_to_llama_tool_param(bad_tool)


# Test multimodal content handling in _generate
def test_generate_with_multimodal_content(mock_llama_client):
    """Test handling of multimodal content in the response."""
    model = ChatMetaLlama(client=mock_llama_client, model_name="test-model")
    
    # Set up response with complex content structure
    mock_response = MagicMock()
    mock_response.completion_message = MagicMock()
    
    # Create content that is a list of items with different types
    text_item1 = MagicMock()
    text_item1.type = "text"
    text_item1.text = "Part 1"
    
    text_item2 = MagicMock()
    text_item2.text = "Part 2"  # No type attribute
    
    # A string item in the list
    string_item = "Part 3"
    
    # Dict item with text
    dict_item = {"type": "text", "text": "Part 4"}
    
    mock_response.completion_message.content = [text_item1, text_item2, string_item, dict_item]
    mock_response.completion_message.tool_calls = []
    mock_response.completion_message.stop_reason = "stop"
    mock_response.metrics = []
    
    mock_llama_client.chat.completions.create.return_value = mock_response
    
    result = model._generate([HumanMessage(content="Test multimodal")])
    assert "Part 1 Part 2 Part 3 Part 4" in result.generations[0].message.content


# Test run_manager callbacks
def test_run_manager_callbacks(mock_llama_client):
    """Test proper use of run_manager callbacks."""
    model = ChatMetaLlama(client=mock_llama_client, model_name="test-model")
    
    # Mock run_manager
    run_manager = MagicMock()
    
    # Create stop sequences to trigger warning
    result = model._generate(
        [HumanMessage(content="Test")], 
        stop=["stop"], 
        run_manager=run_manager
    )
    
    # Verify warning was sent to run_manager
    run_manager.on_text.assert_called_once()
    assert "Warning: 'stop' sequences" in run_manager.on_text.call_args[0][0]


# Test error handling during streaming
def test_stream_error_handling(mock_llama_client):
    """Test error handling during streaming."""
    model = ChatMetaLlama(client=mock_llama_client, model_name="test-model")
    
    # Make the API call raise an error
    mock_llama_client.chat.completions.create.side_effect = ValueError("Stream error")
    
    with pytest.raises(ValueError, match="Stream error"):
        list(model._stream([HumanMessage(content="Test")]))


@pytest.mark.asyncio
async def test_astream_error_handling(mock_llama_client):
    """Test error handling during async streaming."""
    model = ChatMetaLlama(client=mock_llama_client, model_name="test-model")
    
    # Mock run_manager
    run_manager = MagicMock()
    run_manager.on_llm_error = AsyncMock()
    
    # Make the API call raise an error
    mock_llama_client.chat.completions.create.side_effect = ValueError("Async stream error")
    
    with pytest.raises(ValueError, match="Async stream error"):
        async for _ in model._astream([HumanMessage(content="Test")], run_manager=run_manager):
            pass
    
    # Since assert_called_once returns None, we can't await it
    assert run_manager.on_llm_error.assert_called_once() is None


# Test async token callback
@pytest.mark.asyncio
async def test_astream_token_callback(mock_llama_client):
    """Test token callbacks during async streaming."""
    model = ChatMetaLlama(client=mock_llama_client, model_name="test-model")
    
    # Mock run_manager
    run_manager = MagicMock()
    run_manager.on_llm_new_token = AsyncMock()
    
    # Set up a streaming response
    mock_chunk = MagicMock(
        type="completion_message_delta",
        delta=MagicMock(content="Hello world")
    )
    
    # Create an async iterator for the stream
    class AsyncStreamMock:
        async def __aiter__(self):
            return self
            
        async def __anext__(self):
            return mock_chunk
            raise StopAsyncIteration
    
    mock_llama_client.chat.completions.create.return_value = AsyncStreamMock()
    
    # Process the stream (will need manual stop since our mock iterator doesn't actually stop)
    chunks = []
    counter = 0
    async for chunk in model._astream([HumanMessage(content="Test")], run_manager=run_manager):
        chunks.append(chunk)
        counter += 1
        if counter > 0:  # Only process one chunk for the test
            break
    
    # Verify token callback was used - but don't try to await the assertion
    assert run_manager.on_llm_new_token.called
    assert run_manager.on_llm_new_token.call_args[0][0] == "Hello world"


# Test tool call generation
def test_generate_with_invalid_tool_call(mock_llama_client):
    """Test handling of malformed tool calls from API."""
    model = ChatMetaLlama(client=mock_llama_client, model_name="test-model")
    
    # Create a response with a malformed tool call (missing function name)
    mock_response = MagicMock()
    tool_call = MagicMock()
    tool_call.id = "call_123"
    tool_call.function = MagicMock()
    # Deliberately remove the name attribute
    del tool_call.function.name
    tool_call.function.arguments = "{}"
    
    mock_response.completion_message = MagicMock(
        content=None,
        tool_calls=[tool_call],
        stop_reason="tool_calls"
    )
    mock_llama_client.chat.completions.create.return_value = mock_response
    
    result = model._generate([HumanMessage(content="Use tool")])
    
    # Check that the tool call was properly handled as invalid
    assert len(result.generations[0].message.invalid_tool_calls) == 1
    assert "Malformed tool call" in result.generations[0].message.invalid_tool_calls[0]["error"]


# Test _agenerate
@pytest.mark.asyncio
async def test_agenerate_run_manager(mock_llama_client):
    """Test _agenerate with run_manager."""
    model = ChatMetaLlama(client=mock_llama_client, model_name="test-model")
    
    # Mock run_manager
    run_manager = MagicMock()
    run_manager.on_text = AsyncMock()
    
    # Mock the response
    mock_response = MagicMock()
    mock_response.completion_message = MagicMock(
        content="Async response",
        tool_calls=[],
        stop_reason="stop"
    )
    mock_response.metrics = []
    
    # Set up async mock
    mock_llama_client.chat.completions.create = AsyncMock(return_value=mock_response)
    
    # Test with stop parameter to trigger warning
    result = await model._agenerate(
        [HumanMessage(content="Test async")],
        stop=["stop"],
        run_manager=run_manager
    )
    
    # Verify warning was sent without awaiting the assert
    assert run_manager.on_text.called
    call_args = run_manager.on_text.call_args[0]
    assert len(call_args) > 0
    assert "Warning: 'stop' sequences" in call_args[0]
    
    # Verify result
    assert result.generations[0].message.content == "Async response"


# Test content_str if msg_data.content is None
def test_generate_with_none_content(mock_llama_client):
    """Test handling when completion_message.content is None."""
    model = ChatMetaLlama(client=mock_llama_client, model_name="test-model")
    
    # Set up response with None content
    mock_response = MagicMock()
    mock_response.completion_message = MagicMock(
        content=None,
        tool_calls=[],
        stop_reason="stop"
    )
    mock_response.metrics = []
    
    mock_llama_client.chat.completions.create.return_value = mock_response
    
    result = model._generate([HumanMessage(content="Test none content")])
    
    # Content should be an empty string, not None
    assert result.generations[0].message.content == ""


# Test _astream with async iterator that has a non-awaitable __aiter__
@pytest.mark.asyncio
async def test_astream_with_non_awaitable_aiter(mock_llama_client):
    """Test async streaming with an iterator that has a non-awaitable __aiter__."""
    model = ChatMetaLlama(client=mock_llama_client, model_name="test-model")
    
    # Create a custom AsyncIterator class with non-awaitable __aiter__
    class CustomAsyncIterator:
        def __init__(self):
            self.chunks = [
                MagicMock(
                    type="completion_message_delta",
                    delta=MagicMock(content="Test chunk")
                )
            ]
            self.index = 0
        
        def __aiter__(self):
            # Non-awaitable __aiter__ that returns self directly
            return self
        
        async def __anext__(self):
            if self.index < len(self.chunks):
                chunk = self.chunks[self.index]
                self.index += 1
                return chunk
            raise StopAsyncIteration
    
    # Set the mock to return our custom iterator
    mock_llama_client.chat.completions.create.return_value = CustomAsyncIterator()
    
    # Collect chunks
    chunks = []
    async for chunk in model._astream([HumanMessage(content="Test custom iterator")]):
        chunks.append(chunk)
    
    # Verify we got the expected chunk
    assert len(chunks) == 1
    assert chunks[0].message.content == "Test chunk"


# --- Tests for _lc_message_to_llama_message_param coverage ---

def test_lc_ai_message_tool_call_missing_id():
    """Test AIMessage to Llama converter logic for a tool call missing ID."""
    # We mock the relevant attributes of a BaseMessage to simulate
    # how the converter would see an AIMessage with a tool call having id=None.
    # Note: Standard AIMessage construction likely prevents id=None in its .tool_calls list.
    
    mock_message = MagicMock(spec=BaseMessage)
    mock_message.content = "" # Need to define content attribute for the mock
    # Simulate the structure the converter expects for tool_calls. 
    # The converter iterates message.tool_calls, expecting dict-like items or ToolCall objects.
    # Let's mock it yielding ToolCall objects (named tuples) as that's more accurate.
    from langchain_core.messages.tool import ToolCall
    mock_message.tool_calls = [
        ToolCall(name="test_tool", args={"a": 1}, id=None) # id is None
    ]
    # Make it look like an AIMessage for isinstance check
    # A cleaner way might be to patch isinstance if the function uses it heavily,
    # but checking type attribute or similar can also work.
    # Let's assume the function relies on the object structure primarily.

    # Need to ensure isinstance(mock_message, AIMessage) works if used.
    # Patching isinstance or making the mock inherit from AIMessage are options.
    # Easiest: Let's assume the converter branches on type attribute or similar, 
    # or just accesses .tool_calls directly if it's an AIMessage.
    # Re-checking converter: it uses isinstance(message, AIMessage).
    
    # Let's patch isinstance for this test
    with patch("integration.chat_meta_llama.isinstance") as mock_isinstance:
        # Configure mock_isinstance to return True when checking for AIMessage
        # and False otherwise (or use wraps=isinstance for default behavior)
        def side_effect(obj, cls):
            if cls == AIMessage:
                return obj == mock_message # Only return True for our specific mock
            return isinstance(obj, cls) # Default behavior for other types
        mock_isinstance.side_effect = side_effect

        converted = _lc_message_to_llama_message_param(mock_message)
    
    assert converted["role"] == "assistant"
    assert len(converted["tool_calls"]) == 1
    # Check that the converter generated a fallback ID like "tc_0"
    assert converted["tool_calls"][0]["id"].startswith("tc_0")
    assert converted["tool_calls"][0]["function"]["name"] == "test_tool"
    assert converted["tool_calls"][0]["function"]["arguments"] == '{"a": 1}'

# test_lc_ai_message_tool_call_non_str_args: 
# This case is likely prevented by AIMessage Pydantic validation, 
# which expects tool_calls[n].args to be a dict.
# The converter line `tc_args_str = str(tc_args_str)` is a fallback.
# If AIMessage always provides args as a dict (or string via model_dump_json), this branch is less critical.

# test_lc_ai_message_content_conversion_failure:
# AIMessage content is Union[str, List[Union[str, Dict]]]. 
# Pydantic validation on AIMessage init likely prevents NonStringifiable content directly.
# The converter code `except Exception: content_value = ""` is a very broad fallback.

# test_lc_tool_message_missing_id:
# ToolMessage Pydantic validation enforces tool_call_id via @model_validator.
# The converter's check `if not hasattr(message, "tool_call_id")` might be for 
# older/differently constructed ToolMessage-like objects, not standard ones.

# test_lc_tool_message_content_conversion_failure:
# ToolMessage Pydantic validation coerces content to str or raises during init.
# The converter's `except Exception as e: raise ValueError(...)` for content conversion
# is a fallback if the input `message.content` is somehow problematic after Pydantic init.


# --- End tests for _lc_message_to_llama_message_param coverage ---

# --- Tests for _lc_tool_to_llama_tool_param coverage ---

def test_lc_tool_with_non_string_name_desc():
    """Test _lc_tool_to_llama_tool_param with non-string name/description."""
    mock_tool = MagicMock()
    mock_tool.name = 123 # Non-string name
    mock_tool.description = object() # Non-string description
    mock_tool.args_schema = None # To satisfy other conditions

    with pytest.raises(ValueError, match="LangChain tool must have string attributes 'name' and 'description'"):
        _lc_tool_to_llama_tool_param(mock_tool)

def test_lc_tool_with_dict_args_schema():
    """Test _lc_tool_to_llama_tool_param with args_schema as a dict."""
    mock_tool = MagicMock()
    mock_tool.name = "dict_args_tool"
    mock_tool.description = "A tool with dict schema"
    dict_schema = {
        "type": "object",
        "properties": {"param1": {"type": "string"}}
    }
    mock_tool.args_schema = dict_schema

    result = _lc_tool_to_llama_tool_param(mock_tool)
    assert result["function"]["parameters"] == dict_schema

# --- End tests for _lc_tool_to_llama_tool_param coverage ---


# --- Tests for ChatMetaLlama._generate and _agenerate coverage ---

def test_generate_with_failing_tool_conversion(mock_llama_client):
    """Test _generate when a tool passed in kwargs fails conversion."""
    model = ChatMetaLlama(client=mock_llama_client, model_name="test-model")
    messages = [HumanMessage(content="Hello")]
    
    # Create a tool that will fail in _lc_tool_to_llama_tool_param
    bad_tool = MagicMock()
    delattr(bad_tool, 'name') # Missing name attribute causes ValueError
    
    with pytest.raises(ValueError, match="Error converting tools for Llama API"):
        model._generate(messages, tools=[bad_tool])

def test_generate_with_malformed_api_tool_call(mock_llama_client):
    """Test _generate with a tool call from API that has a missing/malformed function name."""
    model = ChatMetaLlama(client=mock_llama_client, model_name="test-model")
    messages = [HumanMessage(content="Use a tool")]

    mock_response = MagicMock(spec=CreateChatCompletionResponse)
    
    # Define metrics on the mock_response itself, as per CreateChatCompletionResponse spec
    mock_response.metrics = [] # Can be None or a list of metric objects
    
    # Case 1: tc.function is None
    tool_call_no_func = MagicMock()
    tool_call_no_func.id = "call_no_func"
    tool_call_no_func.function = None # Malformed

    # Case 2: tc.function.name is None
    tool_call_no_name = MagicMock()
    tool_call_no_name.id = "call_no_name"
    tool_call_no_name.function = MagicMock(name=None, arguments="{}") # Malformed

    # Case 3: tc.function.name is not a string
    tool_call_non_str_name = MagicMock()
    tool_call_non_str_name.id = "call_non_str_name"
    tool_call_non_str_name.function = MagicMock(name=123, arguments="{}") # Malformed

    mock_response.completion_message = MagicMock(
        content=None,
        tool_calls=[tool_call_no_func, tool_call_no_name, tool_call_non_str_name],
        stop_reason="tool_calls"
        # metrics is part of CreateChatCompletionResponse, not completion_message
    )
    mock_llama_client.chat.completions.create.return_value = mock_response

    result = model._generate(messages)
    assert len(result.generations[0].message.tool_calls) == 0 # No valid tool calls
    assert len(result.generations[0].message.invalid_tool_calls) == 3
    assert result.generations[0].message.invalid_tool_calls[0]["id"] == "call_no_func"
    assert "Malformed tool call structure from API" in result.generations[0].message.invalid_tool_calls[0]["error"]
    assert result.generations[0].message.invalid_tool_calls[1]["id"] == "call_no_name"
    assert "Malformed tool call structure from API" in result.generations[0].message.invalid_tool_calls[1]["error"]
    assert result.generations[0].message.invalid_tool_calls[2]["id"] == "call_non_str_name"
    assert "Malformed tool call structure from API" in result.generations[0].message.invalid_tool_calls[2]["error"]


# --- End tests for ChatMetaLlama._generate and _agenerate coverage ---


@pytest.mark.asyncio
async def test_agenerate_run_manager(mock_llama_client):
    """Test _agenerate with run_manager."""
    model = ChatMetaLlama(client=mock_llama_client, model_name="test-model")
    
    # Mock run_manager
    run_manager = MagicMock()
    run_manager.on_text = AsyncMock()
    
    # Mock the response
    mock_response = MagicMock()
    mock_response.completion_message = MagicMock(
        content="Async response",
        tool_calls=[],
        stop_reason="stop"
    )
    mock_response.metrics = []
    
    # Set up async mock
    mock_llama_client.chat.completions.create = AsyncMock(return_value=mock_response)
    
    # Test with stop parameter to trigger warning
    result = await model._agenerate(
        [HumanMessage(content="Test async")],
        stop=["stop"],
        run_manager=run_manager
    )
    
    # Verify warning was sent without awaiting the assert
    assert run_manager.on_text.called
    call_args = run_manager.on_text.call_args[0]
    assert len(call_args) > 0
    assert "Warning: 'stop' sequences" in call_args[0]
    
    # Verify result
    assert result.generations[0].message.content == "Async response"


# Test content_str if msg_data.content is None
def test_generate_with_none_content(mock_llama_client):
    """Test handling when completion_message.content is None."""
    model = ChatMetaLlama(client=mock_llama_client, model_name="test-model")
    
    # Set up response with None content
    mock_response = MagicMock()
    mock_response.completion_message = MagicMock(
        content=None,
        tool_calls=[],
        stop_reason="stop"
    )
    mock_response.metrics = []
    
    mock_llama_client.chat.completions.create.return_value = mock_response
    
    result = model._generate([HumanMessage(content="Test none content")])
    
    # Content should be an empty string, not None
    assert result.generations[0].message.content == ""


# Test _astream with async iterator that has a non-awaitable __aiter__
@pytest.mark.asyncio
async def test_astream_with_non_awaitable_aiter(mock_llama_client):
    """Test async streaming with an iterator that has a non-awaitable __aiter__."""
    model = ChatMetaLlama(client=mock_llama_client, model_name="test-model")
    
    # Create a custom AsyncIterator class with non-awaitable __aiter__
    class CustomAsyncIterator:
        def __init__(self):
            self.chunks = [
                MagicMock(
                    type="completion_message_delta",
                    delta=MagicMock(content="Test chunk")
                )
            ]
            self.index = 0
        
        def __aiter__(self):
            # Non-awaitable __aiter__ that returns self directly
            return self
        
        async def __anext__(self):
            if self.index < len(self.chunks):
                chunk = self.chunks[self.index]
                self.index += 1
                return chunk
            raise StopAsyncIteration
    
    # Set the mock to return our custom iterator
    mock_llama_client.chat.completions.create.return_value = CustomAsyncIterator()
    
    # Collect chunks
    chunks = []
    async for chunk in model._astream([HumanMessage(content="Test custom iterator")]):
        chunks.append(chunk)
    
    # Verify we got the expected chunk
    assert len(chunks) == 1
    assert chunks[0].message.content == "Test chunk"


# --- Tests for _lc_message_to_llama_message_param coverage ---

def test_lc_ai_message_tool_call_missing_id():
    """Test AIMessage to Llama converter logic for a tool call missing ID."""
    # We mock the relevant attributes of a BaseMessage to simulate
    # how the converter would see an AIMessage with a tool call having id=None.
    # Note: Standard AIMessage construction likely prevents id=None in its .tool_calls list.
    
    mock_message = MagicMock(spec=BaseMessage)
    mock_message.content = "" # Need to define content attribute for the mock
    # Simulate the structure the converter expects for tool_calls. 
    # The converter iterates message.tool_calls, expecting dict-like items or ToolCall objects.
    # Let's mock it yielding ToolCall objects (named tuples) as that's more accurate.
    from langchain_core.messages.tool import ToolCall
    mock_message.tool_calls = [
        ToolCall(name="test_tool", args={"a": 1}, id=None) # id is None
    ]
    # Make it look like an AIMessage for isinstance check
    # A cleaner way might be to patch isinstance if the function uses it heavily,
    # but checking type attribute or similar can also work.
    # Let's assume the function relies on the object structure primarily.

    # Need to ensure isinstance(mock_message, AIMessage) works if used.
    # Patching isinstance or making the mock inherit from AIMessage are options.
    # Easiest: Let's assume the converter branches on type attribute or similar, 
    # or just accesses .tool_calls directly if it's an AIMessage.
    # Re-checking converter: it uses isinstance(message, AIMessage).
    
    # Let's patch isinstance for this test
    with patch("integration.chat_meta_llama.isinstance") as mock_isinstance:
        # Configure mock_isinstance to return True when checking for AIMessage
        # and False otherwise (or use wraps=isinstance for default behavior)
        def side_effect(obj, cls):
            if cls == AIMessage:
                return obj == mock_message # Only return True for our specific mock
            return isinstance(obj, cls) # Default behavior for other types
        mock_isinstance.side_effect = side_effect

        converted = _lc_message_to_llama_message_param(mock_message)
    
    assert converted["role"] == "assistant"
    assert len(converted["tool_calls"]) == 1
    # Check that the converter generated a fallback ID like "tc_0"
    assert converted["tool_calls"][0]["id"].startswith("tc_0")
    assert converted["tool_calls"][0]["function"]["name"] == "test_tool"
    assert converted["tool_calls"][0]["function"]["arguments"] == '{"a": 1}'

# test_lc_ai_message_tool_call_non_str_args: 
# This case is likely prevented by AIMessage Pydantic validation, 
# which expects tool_calls[n].args to be a dict.
# The converter line `tc_args_str = str(tc_args_str)` is a fallback.
# If AIMessage always provides args as a dict (or string via model_dump_json), this branch is less critical.

# test_lc_ai_message_content_conversion_failure:
# AIMessage content is Union[str, List[Union[str, Dict]]]. 
# Pydantic validation on AIMessage init likely prevents NonStringifiable content directly.
# The converter code `except Exception: content_value = ""` is a very broad fallback.

# test_lc_tool_message_missing_id:
# ToolMessage Pydantic validation enforces tool_call_id via @model_validator.
# The converter's check `if not hasattr(message, "tool_call_id")` might be for 
# older/differently constructed ToolMessage-like objects, not standard ones.

# test_lc_tool_message_content_conversion_failure:
# ToolMessage Pydantic validation coerces content to str or raises during init.
# The converter's `except Exception as e: raise ValueError(...)` for content conversion
# is a fallback if the input `message.content` is somehow problematic after Pydantic init.


# --- End tests for _lc_message_to_llama_message_param coverage ---

# --- Tests for ChatMetaLlama._stream and _astream coverage ---

def test_stream_with_failing_tool_conversion(mock_llama_client):
    """Test _stream when a tool passed in kwargs fails conversion."""
    model = ChatMetaLlama(client=mock_llama_client, model_name="test-model")
    messages = [HumanMessage(content="Hello")]
    bad_tool = MagicMock()
    delattr(bad_tool, 'name') # Missing name attribute
    
    with pytest.raises(ValueError, match="Error converting tools for Llama API"):
        list(model._stream(messages, tools=[bad_tool]))

def test_stream_with_client_iteration_error(mock_llama_client):
    """Test _stream when the client's stream call raises an error during iteration."""
    model = ChatMetaLlama(client=mock_llama_client, model_name="test-model")
    messages = [HumanMessage(content="Hello")]

    # Make the initial create call work, but iteration fail
    mock_iterator = MagicMock()
    mock_iterator.__iter__.side_effect = RuntimeError("Client stream iteration failed")
    mock_llama_client.chat.completions.create.return_value = mock_iterator
    
    run_manager_mock = MagicMock()
    with pytest.raises(RuntimeError, match="Client stream iteration failed"):
        list(model._stream(messages, run_manager=run_manager_mock))
    run_manager_mock.on_llm_error.assert_called_once()

def test_stream_tool_chunk_missing_id(mock_llama_client):
    """Test _stream with a tool call chunk missing an ID."""
    model = ChatMetaLlama(client=mock_llama_client, model_name="test-model")
    messages = [HumanMessage(content="Tool call")]

    # Chunk with tool call that has a None ID
    # The CreateChatCompletionResponseStreamChunk type hints suggest id is required.
    # We craft a MagicMock to simulate this scenario.
    mock_tc_delta = MagicMock()
    mock_tc_delta.id = None # Falsey ID
    mock_tc_delta.function = MagicMock(name="test_tool", arguments="{}")
    
    stream_chunk = MagicMock( # spec=CreateChatCompletionResponseStreamChunk if it helps
        type="tool_calls_generation_chunk",
        tool_calls=[mock_tc_delta]
    )
    mock_llama_client.chat.completions.create.return_value = iter([stream_chunk])
    
    chunks = list(model._stream(messages))
    assert len(chunks) == 1
    tool_call_chunk_data = chunks[0].message.tool_call_chunks[0]
    assert tool_call_chunk_data["id"].startswith("tc_0") # Default ID assigned

def test_stream_tool_chunk_none_arguments(mock_llama_client):
    """Test _stream with a tool call chunk where arguments is None."""
    model = ChatMetaLlama(client=mock_llama_client, model_name="test-model")
    messages = [HumanMessage(content="Tool call")]

    mock_tc_delta = MagicMock()
    mock_tc_delta.id = "tid1"
    mock_tc_delta.function = MagicMock(name="test_tool", arguments=None) # Arguments is None
    
    stream_chunk = MagicMock(
        type="tool_calls_generation_chunk",
        tool_calls=[mock_tc_delta]
    )
    mock_llama_client.chat.completions.create.return_value = iter([stream_chunk])
    
    chunks = list(model._stream(messages))
    assert len(chunks) == 1
    tool_call_chunk_data = chunks[0].message.tool_call_chunks[0]
    assert tool_call_chunk_data["args"] == "" # Should be empty string if None

def test_stream_final_chunk_no_metrics(mock_llama_client):
    """Test _stream with a final chunk that has no metrics attribute or empty metrics."""
    model = ChatMetaLlama(client=mock_llama_client, model_name="test-model")
    messages = [HumanMessage(content="Finalize")]

    # Scenario 1: metrics attribute is missing (or None)
    final_chunk_no_metrics_attr = MagicMock(
        type="completion_message_stop",
        stop_reason="stop"
    )
    # Ensure 'metrics' attribute doesn't exist or mock it to raise AttributeError if accessed before check
    # For MagicMock, if not set, hasattr will be False by default for arbitrary names.
    # Or, explicitly make it None if the code path distinguishes None from missing.
    # The code uses `if hasattr(chunk, 'metrics') and chunk.metrics:`
    # So, if hasattr is false, it skips. Let's test that.
    # We can also test with chunk.metrics = None or chunk.metrics = []

    mock_llama_client.chat.completions.create.return_value = iter([final_chunk_no_metrics_attr])
    chunks = list(model._stream(messages))
    assert len(chunks) == 1
    assert chunks[0].generation_info is not None
    assert "usage_metadata" not in chunks[0].generation_info

    # Scenario 2: metrics attribute exists but is an empty list
    final_chunk_empty_metrics = MagicMock(
        type="completion_message_stop",
        stop_reason="length",
        metrics=[] # Empty metrics list
    )
    mock_llama_client.chat.completions.create.return_value = iter([final_chunk_empty_metrics])
    chunks = list(model._stream(messages))
    assert len(chunks) == 1
    assert chunks[0].generation_info is not None
    assert "usage_metadata" not in chunks[0].generation_info # or it's an empty dict

@pytest.mark.asyncio
async def test_astream_with_failing_tool_conversion(mock_llama_client):
    """Test _astream when a tool passed in kwargs fails conversion."""
    model = ChatMetaLlama(client=mock_llama_client, model_name="test-model")
    messages = [HumanMessage(content="Hello")]
    bad_tool = MagicMock()
    delattr(bad_tool, 'name')
    
    with pytest.raises(ValueError, match="Error converting tools for Llama API"):
        async for _ in model._astream(messages, tools=[bad_tool]):
            pass

@pytest.mark.asyncio
async def test_astream_with_client_iteration_error(mock_llama_client):
    """Test _astream when the client's async stream raises an error during iteration."""
    model = ChatMetaLlama(client=mock_llama_client, model_name="test-model")
    messages = [HumanMessage(content="Hello")]

    # Make the async iterator raise an error during iteration
    async def failing_iterator():
        yield MagicMock(type="completion_message_delta", delta=MagicMock(content="OK"))
        raise RuntimeError("Async client stream iteration failed")
        yield # Need yield for it to be an async generator

    # Set the create method to return our failing iterator
    mock_llama_client.chat.completions.create = AsyncMock(return_value=failing_iterator())
    
    run_manager_mock = MagicMock()
    run_manager_mock.on_llm_error = AsyncMock() # Mock the async error callback
    # Ensure relevant callbacks used *before* the error are also async if awaited
    run_manager_mock.on_llm_new_token = AsyncMock()

    with pytest.raises(RuntimeError, match="Async client stream iteration failed"):
        async for _ in model._astream(messages, run_manager=run_manager_mock):
            pass
            
    assert run_manager_mock.on_llm_error.call_count == 1 # Check call count, not await result

@pytest.mark.asyncio
async def test_astream_tool_chunk_missing_id(mock_llama_client):
    """Test _astream with a tool call chunk missing an ID."""
    model = ChatMetaLlama(client=mock_llama_client, model_name="test-model")
    messages = [HumanMessage(content="Tool call")]

    mock_tc_delta = MagicMock()
    mock_tc_delta.id = None
    mock_tc_delta.function = MagicMock(name="test_tool", arguments="{}")
    
    stream_chunk = MagicMock(
        type="tool_calls_generation_chunk",
        tool_calls=[mock_tc_delta]
    )

    # Need an async iterator
    async def async_iter():
        yield stream_chunk
        
    mock_llama_client.chat.completions.create = AsyncMock(return_value=async_iter())
    
    chunks = []
    async for chunk in model._astream(messages):
        chunks.append(chunk)
        
    assert len(chunks) == 1
    tool_call_chunk_data = chunks[0].message.tool_call_chunks[0]
    assert tool_call_chunk_data["id"].startswith("tc_0")

@pytest.mark.asyncio
async def test_astream_tool_chunk_none_arguments(mock_llama_client):
    """Test _astream with a tool call chunk where arguments is None."""
    model = ChatMetaLlama(client=mock_llama_client, model_name="test-model")
    messages = [HumanMessage(content="Tool call")]

    mock_tc_delta = MagicMock()
    mock_tc_delta.id = "tid_async"
    mock_tc_delta.function = MagicMock(name="test_tool", arguments=None)
    
    stream_chunk = MagicMock(
        type="tool_calls_generation_chunk",
        tool_calls=[mock_tc_delta]
    )

    async def async_iter():
        yield stream_chunk
        
    mock_llama_client.chat.completions.create = AsyncMock(return_value=async_iter())
    
    chunks = []
    async for chunk in model._astream(messages):
        chunks.append(chunk)

    assert len(chunks) == 1
    tool_call_chunk_data = chunks[0].message.tool_call_chunks[0]
    assert tool_call_chunk_data["args"] == ""

@pytest.mark.asyncio
async def test_astream_final_chunk_no_metrics(mock_llama_client):
    """Test _astream final chunk with missing/empty metrics."""
    model = ChatMetaLlama(client=mock_llama_client, model_name="test-model")
    messages = [HumanMessage(content="Finalize async")]

    # Scenario 1: metrics missing
    final_chunk_no_metrics_attr = MagicMock(
        type="completion_message_stop",
        stop_reason="stop"
    )
    # Remove metrics if it exists from default mock setup
    if hasattr(final_chunk_no_metrics_attr, 'metrics'):
        delattr(final_chunk_no_metrics_attr, 'metrics')

    async def async_iter_1():
        yield final_chunk_no_metrics_attr

    mock_llama_client.chat.completions.create = AsyncMock(return_value=async_iter_1())
    chunks = []
    async for chunk in model._astream(messages):
        chunks.append(chunk)
        
    assert len(chunks) == 1
    assert chunks[0].generation_info is not None
    assert "usage_metadata" not in chunks[0].generation_info

    # Scenario 2: metrics is empty list
    final_chunk_empty_metrics = MagicMock(
        type="completion_message_stop",
        stop_reason="length",
        metrics=[]
    )
    async def async_iter_2():
        yield final_chunk_empty_metrics

    mock_llama_client.chat.completions.create = AsyncMock(return_value=async_iter_2())
    chunks = []
    async for chunk in model._astream(messages):
        chunks.append(chunk)
        
    assert len(chunks) == 1
    assert chunks[0].generation_info is not None
    assert "usage_metadata" not in chunks[0].generation_info


# --- End tests for ChatMetaLlama._stream and _astream coverage ---

# --- Tests for Validators and Identifying Params ---

def test_model_name_validation_warning():
    """Test that the validator function issues a warning for unknown model names."""
    # Test the validator function directly, as Pydantic instantiation 
    # might not trigger warnings reliably in a test context.
    with pytest.warns(UserWarning, match="Model 'unknown-model' is not in the list of known Llama models"):
        # We call the class method validator directly
        ChatMetaLlama.validate_model_name("unknown-model")

def test_identifying_params_with_defaults():
    """Test _identifying_params when params are default (or None)."""
    # Defaults are None for temp, max_tokens, rep_penalty
    model = ChatMetaLlama(
        client=MagicMock(spec=LlamaAPIClient),
        model_name="test-model"
    )
    params = model._identifying_params
    # When default is None, it should appear as None
    assert params == {
        "model_name": "test-model",
        "temperature": None,
        "max_completion_tokens": None,
        "repetition_penalty": None,
    }

def test_identifying_params_with_explicit_values():
    """Test _identifying_params with explicitly set values."""
    model = ChatMetaLlama(
        client=MagicMock(spec=LlamaAPIClient),
        model_name="test-model",
        temperature=0.1,
        max_tokens=50,
        repetition_penalty=1.0
    )
    params = model._identifying_params
    assert params == {
        "model_name": "test-model",
        "temperature": 0.1,
        "max_completion_tokens": 50,
        "repetition_penalty": 1.0,
    }

# Explicit tests for __init__ validation branches might be redundant 
# if test_parameter_validation_edge_cases and test_invalid_parameter_values cover them.
# Let's rely on those for now and revisit if coverage report still shows gaps here.

# --- End Tests for Validators and Identifying Params --- 