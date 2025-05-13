"""Test chat model integration."""

import json
import logging
from datetime import datetime
from unittest.mock import MagicMock, patch

import pytest
from langchain_core.messages import HumanMessage, AIMessage
from llama_api_client import (
    AsyncLlamaAPIClient as OriginalAsyncLlamaAPIClient,  # Import for spec
)

# Import the real class from its actual location BEFORE it might be patched in a fixture
from llama_api_client import LlamaAPIClient as OriginalLlamaAPIClient
from llama_api_client.types import CreateChatCompletionResponse

# Types for Llama API RESPONSE structure
from llama_api_client.types.completion_message import (
    CompletionMessage as LlamaCompletionMessage,
)
from llama_api_client.types.completion_message import (
    ToolCall as LlamaToolCall,  # RESPONSE tool call
)
from llama_api_client.types.completion_message import (
    ToolCallFunction as LlamaToolCallFunction,  # RESPONSE tool call function part
)

# from llama_api_client.types.chat.chat_completion_response_choice import ChatCompletionResponseChoice # This might not be needed if we directly mock CompletionMessage
from pydantic import BaseModel, Field

from langchain_meta import ChatMetaLlama
from langchain_meta.chat_models import (
    LLAMA_DEFAULT_MODEL_NAME,
)
from langchain_meta.chat_meta_llama.serialization import (
    _lc_tool_to_llama_tool_param,
    _normalize_tool_call,
)


# Define MockClients at the module level so it can be used as a type hint
class MockClients:
    def __init__(self, sync_client, async_client, sync_constructor, async_constructor):
        self.sync = sync_client
        self.async_client = async_client
        self.sync_constructor = sync_constructor
        self.async_constructor = async_constructor


@pytest.fixture
def mock_llama_client_fixture():
    # Define a common mock response object factory here to ensure consistency
    def create_mock_response_object():
        mock_response_obj = MagicMock(spec=CreateChatCompletionResponse)

        mock_completion_msg = MagicMock(spec=LlamaCompletionMessage)
        mock_completion_msg.content = {
            "type": "text",
            "text": "Mocked test response",
        }  # Content can be str or MessageTextContentItem

        mock_tool_call_func = MagicMock(spec=LlamaToolCallFunction)
        mock_tool_call_func.name = "mock_tool_name"
        mock_tool_call_func.arguments = json.dumps({"arg1": "val1", "arg2": 2})

        mock_tool_call = MagicMock(spec=LlamaToolCall)
        mock_tool_call.id = "tool_xyz789"
        # mock_tool_call.type = "function" # LlamaToolCall doesn't have a 'type' field, it's implicit by having a 'function' attr
        mock_tool_call.function = mock_tool_call_func

        # Ensure the mock_tool_call can be .to_dict()-ed if our main code tries that on items
        mock_tool_call.to_dict = MagicMock(
            return_value={
                "id": mock_tool_call.id,
                "function": {
                    "name": mock_tool_call_func.name,
                    "arguments": mock_tool_call_func.arguments,
                },
            }
        )

        mock_completion_msg.tool_calls = [mock_tool_call]
        mock_completion_msg.stop_reason = "stop"
        mock_completion_msg.role = "assistant"
        mock_completion_msg.to_dict = MagicMock(
            return_value={
                "role": "assistant",
                "content": mock_completion_msg.content,
                "tool_calls": [tc.to_dict() for tc in mock_completion_msg.tool_calls],
                "stop_reason": mock_completion_msg.stop_reason,
            }
        )
        mock_response_obj.completion_message = mock_completion_msg
        # Mock metrics
        mock_metric_prompt = MagicMock()
        mock_metric_prompt.metric = "num_prompt_tokens"
        mock_metric_prompt.value = 20
        mock_metric_completion = MagicMock()
        mock_metric_completion.metric = "num_completion_tokens"
        mock_metric_completion.value = 40
        mock_metric_total = MagicMock()
        mock_metric_total.metric = "num_total_tokens"
        mock_metric_total.value = 60
        for m in [mock_metric_prompt, mock_metric_completion, mock_metric_total]:
            m.to_dict = MagicMock(return_value={"metric": m.metric, "value": m.value})
        mock_response_obj.metrics = [
            mock_metric_prompt,
            mock_metric_completion,
            mock_metric_total,
        ]
        mock_response_obj.x_request_id = "mock_req_id_abc"
        mock_response_obj.to_dict = MagicMock(
            return_value={
                "completion_message": mock_completion_msg.to_dict(),
                "metrics": [m.to_dict() for m in mock_response_obj.metrics],
                "x_request_id": mock_response_obj.x_request_id,
            }
        )
        return mock_response_obj

    with (
        patch("llama_api_client.LlamaAPIClient") as mock_sync_client_constructor,
        patch("llama_api_client.AsyncLlamaAPIClient") as mock_async_client_constructor,
    ):
        # Setup mock for Sync Client
        mock_sync_instance = MagicMock(spec=OriginalLlamaAPIClient)
        mock_sync_instance.__class__ = OriginalLlamaAPIClient
        sync_chat_attr = MagicMock()
        sync_completions_attr = MagicMock()
        # Use a factory to get a fresh response object for each call if needed, or a shared one
        sync_create_method_mock = MagicMock(
            name="sync_create_method_mock", return_value=create_mock_response_object()
        )

        # NEW: Side effect for sync streaming
        def sync_create_side_effect(*args, **kwargs):
            if kwargs.get("stream"):
                # Return an ITERATOR of mock chunks for streaming
                mock_chunk1_completion_msg = MagicMock(spec=LlamaCompletionMessage)
                mock_chunk1_completion_msg.content = "Streamed part 1"
                mock_chunk1_completion_msg.tool_calls = None
                mock_chunk1_completion_msg.stop_reason = None

                mock_chunk1 = MagicMock(
                    spec=CreateChatCompletionResponse
                )  # Assuming stream yields these types
                mock_chunk1.completion_message = mock_chunk1_completion_msg
                mock_chunk1.usage = None
                mock_chunk1.x_request_id = "stream_req_id_1"
                mock_chunk1.to_dict = MagicMock(
                    return_value={
                        "completion_message": {
                            "content": "Streamed part 1",
                            "tool_calls": None,
                            "stop_reason": None,
                        },
                        "x_request_id": "stream_req_id_1",
                    }
                )

                mock_chunk2_completion_msg = MagicMock(spec=LlamaCompletionMessage)
                mock_chunk2_completion_msg.content = " part 2"

                # Example of a tool call appearing in a stream chunk
                mock_stream_tool_func = MagicMock(spec=LlamaToolCallFunction)
                mock_stream_tool_func.name = "stream_tool"
                mock_stream_tool_func.arguments = (
                    '{"param":"value"}'  # Corrected JSON string
                )

                mock_stream_tool_call = MagicMock(spec=LlamaToolCall)
                mock_stream_tool_call.id = "stream_tool_id_1"
                mock_stream_tool_call.function = mock_stream_tool_func
                mock_stream_tool_call.to_dict = MagicMock(
                    return_value={
                        "id": "stream_tool_id_1",
                        "function": {
                            "name": "stream_tool",
                            "arguments": '{"param":"value"}',
                        },  # Corrected JSON string
                    }
                )
                mock_chunk2_completion_msg.tool_calls = [mock_stream_tool_call]
                mock_chunk2_completion_msg.stop_reason = "tool_calls"

                mock_chunk2 = MagicMock(spec=CreateChatCompletionResponse)
                mock_chunk2.completion_message = mock_chunk2_completion_msg
                mock_chunk2.usage = (
                    None  # Usage might appear at the end or not at all in chunks
                )
                mock_chunk2.x_request_id = "stream_req_id_2"
                mock_chunk2.to_dict = MagicMock(
                    return_value={
                        "completion_message": {
                            "content": " part 2",
                            "tool_calls": [mock_stream_tool_call.to_dict()],
                            "stop_reason": "tool_calls",
                        },
                        "x_request_id": "stream_req_id_2",
                    }
                )

                # Add a final chunk if necessary, e.g., with stop reason and usage
                mock_final_chunk_completion_msg = MagicMock(spec=LlamaCompletionMessage)
                mock_final_chunk_completion_msg.content = (
                    None  # Often empty in final chunk
                )
                mock_final_chunk_completion_msg.tool_calls = None
                mock_final_chunk_completion_msg.stop_reason = "stop"

                mock_final_chunk = MagicMock(spec=CreateChatCompletionResponse)
                mock_final_chunk.completion_message = mock_final_chunk_completion_msg
                # Mock final usage if applicable to your API's streaming
                mock_final_usage = MagicMock()
                mock_final_usage.prompt_tokens = 10
                mock_final_usage.completion_tokens = 5
                mock_final_usage.total_tokens = 15
                mock_final_usage.to_dict = MagicMock(
                    return_value=mock_final_usage.__dict__
                )
                mock_final_chunk.usage = mock_final_usage
                mock_final_chunk.x_request_id = "stream_req_id_final"
                mock_final_chunk.to_dict = MagicMock(
                    return_value={
                        "completion_message": {
                            "content": None,
                            "tool_calls": None,
                            "stop_reason": "stop",
                        },
                        "usage": mock_final_usage.to_dict(),
                        "x_request_id": "stream_req_id_final",
                    }
                )

                return iter([mock_chunk1, mock_chunk2, mock_final_chunk])
            else:
                # Default non-streaming behavior
                return create_mock_response_object()

        sync_create_method_mock.side_effect = (
            sync_create_side_effect  # Apply the side effect
        )

        sync_completions_attr.create = sync_create_method_mock
        sync_chat_attr.completions = sync_completions_attr
        mock_sync_instance.chat = sync_chat_attr
        mock_sync_client_constructor.return_value = mock_sync_instance

        # Setup mock for Async Client
        mock_async_instance = MagicMock(spec=OriginalAsyncLlamaAPIClient)
        mock_async_instance.__class__ = OriginalAsyncLlamaAPIClient  # Masquerade
        async_chat_attr = MagicMock()
        async_completions_attr = MagicMock()

        # Async side_effect function for the async client's create method
        async def async_mock_create_side_effect(*args, **kwargs):
            return create_mock_response_object()  # Return a fresh mock response

        async_create_method_mock = MagicMock(
            name="async_create_method_mock", side_effect=async_mock_create_side_effect
        )
        async_completions_attr.create = (
            async_create_method_mock  # Assign the async mock
        )
        async_chat_attr.completions = async_completions_attr
        mock_async_instance.chat = async_chat_attr
        mock_async_client_constructor.return_value = mock_async_instance

        # MockClients class is now defined at module level
        yield MockClients(
            sync_client=mock_sync_instance,
            async_client=mock_async_instance,
            sync_constructor=mock_sync_client_constructor,
            async_constructor=mock_async_client_constructor,
        )


@pytest.fixture
def configured_mock_llama_clients(mock_llama_client_fixture):  # Renamed for clarity
    # This fixture now provides the MockClients object containing both client instances
    return mock_llama_client_fixture


def test_chat_meta_llama_initialization(
    configured_mock_llama_clients: MockClients,
):  # Updated fixture name
    # Test initialization with only api_key (creates real clients, but uses mocked constructors if they were used internally)
    llm_only_api_key = ChatMetaLlama(api_key="test-key")
    assert llm_only_api_key.model_name == LLAMA_DEFAULT_MODEL_NAME
    if llm_only_api_key.client is not None:
        assert isinstance(llm_only_api_key.client, OriginalLlamaAPIClient)
        # We can't easily assert it used the mock_sync_client_constructor here if it created a real one,
        # unless we check call_count on configured_mock_llama_clients.sync_constructor, but that's complex.
        # The key is that a client IS created.

    # Test initialization when sync client is explicitly passed
    llm_with_sync_client = ChatMetaLlama(
        client=configured_mock_llama_clients.sync, api_key="test-key-ignored"
    )
    assert llm_with_sync_client.client is configured_mock_llama_clients.sync
    # In this case, _ensure_client_initialized will create a real async client using "test-key-ignored"
    # This is okay if we don't call async methods on this instance, or if we want to test that path.

    # Test initialization when async client is explicitly passed
    llm_with_async_client = ChatMetaLlama(
        async_client=configured_mock_llama_clients.async_client,
        api_key="test-key-ignored-too",
    )
    assert (
        llm_with_async_client._async_client
        is configured_mock_llama_clients.async_client
    )

    # Test initialization when BOTH clients are explicitly passed
    llm_with_both_clients = ChatMetaLlama(
        client=configured_mock_llama_clients.sync,
        async_client=configured_mock_llama_clients.async_client,
        api_key="test-key-should-be-fully-ignored",
    )
    assert llm_with_both_clients.client is configured_mock_llama_clients.sync
    assert (
        llm_with_both_clients._async_client
        is configured_mock_llama_clients.async_client
    )
    # Here, _ensure_client_initialized should do nothing as both are provided.


@pytest.mark.asyncio
async def test_agenerate_method(
    configured_mock_llama_clients: MockClients,
):  # Updated fixture name
    # Pass BOTH mocked clients to ensure no real client is created or used
    llm = ChatMetaLlama(
        client=configured_mock_llama_clients.sync,
        async_client=configured_mock_llama_clients.async_client,
        model_name="CustomTestModel",
        api_key="test-key-should-be-ignored",
    )

    messages = [HumanMessage(content="Hello")]
    result = await llm._agenerate(messages)

    assert len(result.generations) == 1
    gen = result.generations[0]
    # Assert content (parsed from dict)
    assert gen.message.content == "Mocked test response"

    # Check for tool calls (parsed from raw Llama format)
    assert len(gen.message.tool_calls) == 1
    parsed_tc = gen.message.tool_calls[0]
    assert parsed_tc["id"] == "tool_xyz789"
    assert parsed_tc["name"] == "mock_tool_name"
    assert parsed_tc["args"] == {"arg1": "val1", "arg2": 2}

    # Check usage metadata (parsed from metrics list)
    assert gen.message.usage_metadata is not None
    assert gen.message.usage_metadata["input_tokens"] == 20
    assert gen.message.usage_metadata["output_tokens"] == 40
    assert gen.message.usage_metadata["total_tokens"] == 60

    # Check generation_info fields
    assert gen.generation_info is not None
    assert (
        gen.generation_info["finish_reason"] == "stop"
    )  # from completion_message.stop_reason
    assert (
        gen.generation_info["x_request_id"] == "mock_req_id_abc"
    )  # from response.x_request_id
    assert "usage_metadata" in gen.generation_info  # From parsed metrics
    assert (
        "response_metadata" in gen.generation_info
    )  # Should contain to_dict() of CreateChatCompletionResponse
    assert gen.generation_info["response_metadata"]["x_request_id"] == "mock_req_id_abc"

    # This is an async test, should assert the async_client's mock
    configured_mock_llama_clients.async_client.chat.completions.create.assert_called_once()
    call_args = (
        configured_mock_llama_clients.async_client.chat.completions.create.call_args
    )
    assert call_args.kwargs["model"] == "CustomTestModel"
    assert call_args.kwargs["messages"] == [{"role": "user", "content": "Hello"}]


# --- Tests for _lc_tool_to_llama_tool_param ---


class MyTestToolArgs(BaseModel):
    """Arguments for my test tool."""

    param1: str = Field(description="First parameter")
    param2: int = Field(description="Second parameter")


class MySimpleTool(BaseModel):
    """A simple Pydantic model tool."""

    query: str = Field(description="The query string")


class NoDocstringTool(BaseModel):
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
            "parameters": MySimpleTool.model_json_schema(),
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
            "parameters": NoDocstringTool.model_json_schema(),
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
            "parameters": MyTestToolArgs.model_json_schema(),
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


def test_tool_choice_parameter_filtering(configured_mock_llama_clients: MockClients):
    """Test that tool_choice parameter is filtered out for sync calls."""
    llm = ChatMetaLlama(
        client=configured_mock_llama_clients.sync,
        async_client=configured_mock_llama_clients.async_client,  # provide async as well for full init
        model_name="TestModel",
        api_key="test-key-should-be-ignored",
    )

    messages = [HumanMessage(content="Hello")]
    # Call _generate with tool_choice
    llm._generate(messages, tool_choice="auto", temperature=0.7, max_tokens=100)

    # Assert the sync client was called
    configured_mock_llama_clients.sync.chat.completions.create.assert_called_once()
    call_args = configured_mock_llama_clients.sync.chat.completions.create.call_args

    # Assert 'tool_choice' is NOT in the API call kwargs
    assert "tool_choice" not in call_args.kwargs
    # Assert other params are still there
    assert "temperature" in call_args.kwargs
    assert call_args.kwargs["temperature"] == 0.7
    assert (
        "max_completion_tokens" in call_args.kwargs
    )  # Check for the Llama API parameter name
    assert (
        call_args.kwargs["max_completion_tokens"] == 100
    )  # Check for the Llama API parameter name
    assert call_args.kwargs["model"] == "TestModel"
    assert call_args.kwargs["messages"] == [{"role": "user", "content": "Hello"}]


@pytest.mark.asyncio
async def test_async_tool_choice_parameter_filtering(
    configured_mock_llama_clients: MockClients,
):
    """Test that tool_choice parameter is filtered out for async calls."""
    llm = ChatMetaLlama(
        client=configured_mock_llama_clients.sync,  # provide sync as well for full init
        async_client=configured_mock_llama_clients.async_client,
        model_name="TestModel",
        api_key="test-key-should-be-ignored",
    )

    messages = [HumanMessage(content="Hello")]
    # Call _agenerate with tool_choice
    await llm._agenerate(
        messages, tool_choice="required", temperature=0.5, max_tokens=200
    )

    # Assert the async client was called
    configured_mock_llama_clients.async_client.chat.completions.create.assert_called_once()
    call_args = (
        configured_mock_llama_clients.async_client.chat.completions.create.call_args
    )

    # Assert 'tool_choice' is NOT in the API call kwargs
    assert "tool_choice" not in call_args.kwargs
    # Assert other params are still there
    assert "temperature" in call_args.kwargs
    assert call_args.kwargs["temperature"] == 0.5
    assert (
        "max_completion_tokens" in call_args.kwargs
    )  # Check for the Llama API parameter name
    assert (
        call_args.kwargs["max_completion_tokens"] == 200
    )  # Check for the Llama API parameter name
    assert call_args.kwargs["model"] == "TestModel"
    assert call_args.kwargs["messages"] == [{"role": "user", "content": "Hello"}]


@pytest.mark.asyncio
async def test_bind_with_tool_choice_async(
    configured_mock_llama_clients: MockClients,
):  # Updated fixture name
    """Test that bind method properly handles tool_choice parameter in async calls."""
    from langchain_meta.chat_models import ChatMetaLlama

    # Create a minimal instance with the already mocked clients
    llm = ChatMetaLlama(
        client=configured_mock_llama_clients.sync,
        async_client=configured_mock_llama_clients.async_client,
        model_name="TestModel",
        api_key="test-key-should-be-ignored",
    )

    # Create a simple tool
    class SimpleToolSchema(BaseModel):
        """A simple test tool."""

        name: str = Field(description="The name to greet")

    # Test binding with tool_choice as object
    bound_llm = llm.bind(
        tools=[SimpleToolSchema],
        tool_choice={"type": "function", "function": {"name": "SimpleToolSchema"}},
    )

    # Verify the binding worked (didn't throw error)
    assert bound_llm is not None
    # Check kwargs of the RunnableBinding object
    assert "tools" in bound_llm.kwargs, "Tools not found in bound_llm.kwargs"
    assert bound_llm.kwargs["tools"] == [SimpleToolSchema], (
        "Incorrect tools in bound_llm.kwargs"
    )
    assert "tool_choice" in bound_llm.kwargs, (
        "tool_choice not found in bound_llm.kwargs"
    )

    # Now call with a message to ensure it works
    messages = [HumanMessage(content="Hello")]

    # Use ainvoke instead of directly calling _agenerate
    response_message = await bound_llm.ainvoke(messages)
    assert isinstance(response_message, AIMessage)  # Ensure we get an AIMessage

    # Verify the API was called with the correct parameters
    configured_mock_llama_clients.async_client.chat.completions.create.assert_called_once()

    call_args = (
        configured_mock_llama_clients.async_client.chat.completions.create.call_args
    )
    # Assert that 'tool_choice' is NOT passed to the Llama API
    assert "tool_choice" not in call_args.kwargs
    # Assert that 'tools' ARE passed to the Llama API
    assert "tools" in call_args.kwargs
    assert len(call_args.kwargs["tools"]) == 1
    assert call_args.kwargs["tools"][0]["function"]["name"] == "SimpleToolSchema"

    # Reset mock for next test
    configured_mock_llama_clients.async_client.chat.completions.create.reset_mock()

    # Test with "none" tool_choice
    bound_llm_none = llm.bind(tools=[SimpleToolSchema], tool_choice="none")
    assert bound_llm_none is not None
    # Check kwargs of the RunnableBinding object for bound_llm_none
    assert "tools" in bound_llm_none.kwargs, (
        "Tools not found in bound_llm_none.kwargs (tool_choice='none')"
    )
    assert bound_llm_none.kwargs["tools"] == [SimpleToolSchema], (
        "Tools incorrect in bound_llm_none.kwargs (tool_choice='none')"
    )
    assert bound_llm_none.kwargs.get("tool_choice") == "none", (
        "tool_choice was not 'none' in bound_llm_none.kwargs"
    )

    # Use ainvoke instead of directly calling _agenerate
    response_message_none = await bound_llm_none.ainvoke(messages)
    assert isinstance(response_message_none, AIMessage)  # Ensure we get an AIMessage

    configured_mock_llama_clients.async_client.chat.completions.create.assert_called_once()
    call_args_none = (
        configured_mock_llama_clients.async_client.chat.completions.create.call_args
    )
    # Assert that 'tool_choice' is NOT passed to the Llama API
    assert "tool_choice" not in call_args_none.kwargs
    # Tools should still be sent, as they are bound.
    assert "tools" in call_args_none.kwargs
    assert len(call_args_none.kwargs["tools"]) == 1
    assert call_args_none.kwargs["tools"][0]["function"]["name"] == "SimpleToolSchema"


def test_stream_method(configured_mock_llama_clients: MockClients):
    llm = ChatMetaLlama(
        client=configured_mock_llama_clients.sync,
        async_client=configured_mock_llama_clients.async_client,  # Keep async for full init
        model_name="TestStreamModel",
        api_key="test-key-streaming-ignored",
    )
    messages = [HumanMessage(content="Stream this for me")]

    chunks = list(llm.stream(messages))  # Consume the iterator

    assert len(chunks) == 3  # Based on the mock_sync_streaming_response side_effect

    # Chunk 1
    assert chunks[0].message.content == "Streamed part 1"
    assert chunks[0].generation_info["x_request_id"] == "stream_req_id_1"
    assert not chunks[0].message.tool_call_chunks  # No tool calls in first chunk

    # Chunk 2
    assert chunks[1].message.content == " part 2"
    assert chunks[1].generation_info["x_request_id"] == "stream_req_id_2"
    assert chunks[1].generation_info["finish_reason"] == "tool_calls"
    assert len(chunks[1].message.tool_call_chunks) == 1
    tc_chunk = chunks[1].message.tool_call_chunks[0]
    assert tc_chunk["id"] == "stream_tool_id_1"
    assert tc_chunk["name"] == "stream_tool"
    assert tc_chunk["args"] == '{"param":"value"}'  # Corrected expected JSON string

    # Chunk 3 (Final)
    assert chunks[2].message.content == ""  # Content might be empty in final chunk
    assert chunks[2].generation_info["x_request_id"] == "stream_req_id_final"
    assert chunks[2].generation_info["finish_reason"] == "stop"
    assert "chunk_usage" in chunks[2].generation_info
    assert chunks[2].generation_info["chunk_usage"]["prompt_tokens"] == 10

    # Verify the sync client's create method was called with stream=True
    configured_mock_llama_clients.sync.chat.completions.create.assert_called_once()
    call_args = configured_mock_llama_clients.sync.chat.completions.create.call_args
    assert call_args.kwargs["stream"] is True
    assert call_args.kwargs["model"] == "TestStreamModel"


def test_tool_calls_serialization():
    """Test that tool calls are properly serialized for LangGraph."""
    # Test the _normalize_tool_call function directly
    tc = {"id": "123", "name": "test_tool", "args": {"key": "value"}}
    normalized = _normalize_tool_call(tc)
    assert normalized["id"] == "123"
    assert normalized["name"] == "test_tool"
    assert normalized["args"] == {"key": "value"}
    assert normalized["type"] == "function"

    # Test with complex object that would fail serialization
    complex_tc = {"id": "123", "name": "test_tool", "args": {"key": datetime.now()}}
    normalized = _normalize_tool_call(complex_tc)
    # Should convert datetime to string or dict
    assert isinstance(normalized["args"]["key"], (str, dict))


@pytest.mark.asyncio
async def test_langgraph_serialization_compatibility(
    configured_mock_llama_clients: MockClients,
):  # Updated fixture name
    """Test that output from Meta LLM can be JSON serialized for LangGraph."""

    # For this test, we are testing ChatMetaLlama's output processing,
    # so we make the mock async_client's create method return a specific structured response.
    mock_structured_response = MagicMock(spec=CreateChatCompletionResponse)
    mock_completion_msg = MagicMock()
    mock_completion_msg.content = "Test response for langgraph"
    mock_raw_tc = MagicMock(
        id="lg_tool_123",
        function=MagicMock(
            name="langgraph_tool", arguments=json.dumps({"lg_key": "lg_value"})
        ),
    )
    mock_completion_msg.tool_calls = [mock_raw_tc]
    mock_completion_msg.stop_reason = "tool_calls"
    mock_structured_response.completion_message = mock_completion_msg
    mock_structured_response.metrics = None  # Keep it simple for this test
    mock_structured_response.x_request_id = "mock_req_langgraph"
    mock_structured_response.to_dict = MagicMock(
        return_value={
            "completion_message": mock_completion_msg.to_dict(),
            "x_request_id": "mock_req_langgraph",
        }
    )

    async def mock_create_for_langgraph_test(*args, **kwargs):
        return mock_structured_response

    # Directly configure the create method of the async mock instance from the fixture
    configured_mock_llama_clients.async_client.chat.completions.create.side_effect = (
        mock_create_for_langgraph_test
    )

    llm = ChatMetaLlama(
        client=configured_mock_llama_clients.sync,  # Provide sync mock too
        async_client=configured_mock_llama_clients.async_client,
        api_key="test-key-ignored-for-langgraph",
    )
    result = await llm._agenerate([HumanMessage(content="test")])

    # Test if result can be JSON serialized (critical for LangGraph)
    try:
        serialized = json.dumps(result, default=str)
        assert serialized  # Ensure we got a valid string
    except TypeError as e:
        pytest.fail(f"Result is not JSON serializable: {e}")


def test_normalize_tool_call_various_cases():
    from langchain_meta.chat_meta_llama.serialization import _normalize_tool_call

    # Case: missing id, args as string
    tc = {"name": "foo", "args": '{"bar": 1}'}
    norm = _normalize_tool_call(tc)
    assert isinstance(norm["id"], str) and norm["id"]
    assert norm["name"] == "foo"
    assert norm["args"] == {"bar": 1}
    assert norm["type"] == "function"

    # Case: args as non-dict, non-string
    tc = {"name": "foo", "args": 123}
    norm = _normalize_tool_call(tc)
    assert norm["args"] == {"value": "123"}

    # Case: missing name
    tc = {"id": "abc", "args": {}}
    norm = _normalize_tool_call(tc)
    assert norm["name"] == "unknown_tool"

    # Case: id present, name present, args already dict
    tc = {"id": "id123", "name": "bar", "args": {"x": 1}}
    norm = _normalize_tool_call(tc)
    assert norm["id"] == "id123"
    assert norm["name"] == "bar"
    assert norm["args"] == {"x": 1}
