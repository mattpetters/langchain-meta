"""Additional tests to increase code coverage for ChatMetaLlama class to 90%."""

from unittest.mock import MagicMock, patch, AsyncMock

import pytest
from langchain_core.messages import (
    AIMessage,
    HumanMessage,
    ToolMessage,
)
from llama_api_client import LlamaAPIClient

from integration.chat_meta_llama import (
    ChatMetaLlama,
    _lc_message_to_llama_message_param,
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


def test_multimodal_content_additional_cases(mock_llama_client):
    """Test additional multimodal content parsing cases in _generate."""
    model = ChatMetaLlama(client=mock_llama_client, model_name="test-model")

    # Test scenario where msg_data.content is an object with a text attribute
    mock_response = MagicMock()
    mock_response.completion_message = MagicMock()

    # Create a content object with a text attribute
    content_obj = MagicMock()
    content_obj.text = "Text from object attribute"

    mock_response.completion_message.content = content_obj
    mock_response.completion_message.tool_calls = []
    mock_response.completion_message.stop_reason = "stop"
    mock_response.metrics = []

    mock_llama_client.chat.completions.create.return_value = mock_response

    result = model._generate([HumanMessage(content="Test multimodal")])

    # Verify that the text from the content object was extracted
    assert result.generations[0].message.content == "Text from object attribute"


def test_stream_with_run_manager_callbacks(mock_llama_client):
    """Test run_manager callbacks in _stream method."""
    model = ChatMetaLlama(client=mock_llama_client, model_name="test-model")

    # Mock run_manager
    run_manager = MagicMock()

    # Set up iterator response for streaming
    chunk1 = MagicMock()
    chunk1.type = "completion_message_delta"
    chunk1.delta = MagicMock()
    chunk1.delta.content = "Hello "

    chunk2 = MagicMock()
    chunk2.type = "completion_message_delta"
    chunk2.delta = MagicMock()
    chunk2.delta.content = "world!"

    mock_llama_client.chat.completions.create.return_value = [chunk1, chunk2]

    # Run the stream method
    chunks = list(
        model._stream([HumanMessage(content="Test")], run_manager=run_manager)
    )

    # Verify callbacks were called for each token
    assert run_manager.on_llm_new_token.call_count == 2
    run_manager.on_llm_new_token.assert_any_call("Hello ")
    run_manager.on_llm_new_token.assert_any_call("world!")

    # Verify chunks were yielded correctly
    assert len(chunks) == 2
    assert chunks[0].message.content == "Hello "
    assert chunks[1].message.content == "world!"


def test_stream_tool_call_with_args_appending(mock_llama_client):
    """Test _stream handling of tool call chunks with args appending."""
    model = ChatMetaLlama(client=mock_llama_client, model_name="test-model")

    # Set up tool call chunks that should be combined
    chunk1 = MagicMock()
    chunk1.type = "tool_calls_generation_chunk"
    tool_call1 = MagicMock()
    tool_call1.id = "tc_1"
    tool_call1.function = MagicMock()
    tool_call1.function.name = "search"
    tool_call1.function.arguments = '{"que'
    chunk1.tool_calls = [tool_call1]

    chunk2 = MagicMock()
    chunk2.type = "tool_calls_generation_chunk"
    tool_call2 = MagicMock()
    tool_call2.id = "tc_1"  # Same ID to continue previous chunk
    tool_call2.function = MagicMock()
    tool_call2.function.name = "search"
    tool_call2.function.arguments = 'ry":"test"}'
    chunk2.tool_calls = [tool_call2]

    mock_llama_client.chat.completions.create.return_value = [chunk1, chunk2]

    # Run the stream method
    chunks = list(model._stream([HumanMessage(content="Test")]))

    # Verify tool calls were processed correctly across chunks
    assert len(chunks) == 2

    # First chunk should have partial args
    assert len(chunks[0].message.tool_call_chunks) == 1
    assert chunks[0].message.tool_call_chunks[0]["args"] == '{"que'

    # Second chunk should also have just its partial args
    # (the combination happens in the current_tool_calls dict internally)
    assert len(chunks[1].message.tool_call_chunks) == 1
    assert chunks[1].message.tool_call_chunks[0]["args"] == 'ry":"test"}'


@pytest.mark.asyncio
async def test_astream_exception_handling(mock_llama_client):
    """Test _astream handling of exceptions when using the async iterator."""
    model = ChatMetaLlama(client=mock_llama_client, model_name="test-model")

    # Set up the mock to raise an exception
    mock_llama_client.chat.completions.create.side_effect = ValueError("Stream error")

    # We should get the ValueError
    with pytest.raises(ValueError, match="Stream error"):
        async for _ in model._astream([HumanMessage(content="Test")]):
            pass


@pytest.mark.asyncio
async def test_astream_run_manager_on_llm_new_token(mock_llama_client):
    """Test run_manager.on_llm_new_token in _astream."""
    model = ChatMetaLlama(client=mock_llama_client, model_name="test-model")

    # Mock run_manager with AsyncMock for async methods
    run_manager = MagicMock()
    run_manager.on_llm_new_token = AsyncMock()

    # Create async iterator that yields content delta chunks
    async def async_iter():
        chunk1 = MagicMock()
        chunk1.type = "completion_message_delta"
        chunk1.delta = MagicMock()
        chunk1.delta.content = "Hello "
        yield chunk1

        chunk2 = MagicMock()
        chunk2.type = "completion_message_delta"
        chunk2.delta = MagicMock()
        chunk2.delta.content = "World!"
        yield chunk2

    # Setup mock to return our async iterator
    mock_llama_client.chat.completions.create.return_value = async_iter()

    # Consume the async iterator
    chunks = []
    async for chunk in model._astream(
        [HumanMessage(content="Test")], run_manager=run_manager
    ):
        chunks.append(chunk)

    # Verify on_llm_new_token was called for each content chunk
    assert run_manager.on_llm_new_token.await_count == 2
    run_manager.on_llm_new_token.assert_any_await("Hello ")
    run_manager.on_llm_new_token.assert_any_await("World!")


@pytest.mark.asyncio
async def test_astream_final_chunk_with_metrics(mock_llama_client):
    """Test _astream handling of metrics in final completion_message_stop chunk."""
    model = ChatMetaLlama(client=mock_llama_client, model_name="test-model")

    # Create mock metrics
    metric1 = MagicMock()
    metric1.metric = "prompt_tokens"
    metric1.value = 10

    metric2 = MagicMock()
    metric2.metric = "completion_tokens"
    metric2.value = 20

    # Create async iterator that yields a final chunk with metrics
    async def async_iter():
        chunk = MagicMock()
        chunk.type = "completion_message_stop"
        chunk.stop_reason = "complete"
        chunk.metrics = [metric1, metric2]
        yield chunk

    # Setup mock to return our async iterator
    mock_llama_client.chat.completions.create.return_value = async_iter()

    # Consume the async iterator
    chunks = []
    async for chunk in model._astream([HumanMessage(content="Test")]):
        chunks.append(chunk)

    # Verify metrics were properly extracted into generation_info
    assert len(chunks) == 1
    assert chunks[0].generation_info is not None
    assert chunks[0].generation_info["finish_reason"] == "complete"
    assert chunks[0].generation_info["usage_metadata"] == {
        "prompt_tokens": 10,
        "completion_tokens": 20,
    }


def test_init_validation_non_number_max_tokens():
    """Test init validation with non-number max_tokens."""
    with pytest.raises(ValueError, match="max_tokens must be at least 1"):
        ChatMetaLlama(
            model_name="test-model",
            max_tokens=0,  # Use 0, which is invalid but a valid number
            client=MagicMock(spec=LlamaAPIClient),
        )


def test_init_validation_invalid_temperature():
    """Test init validation with invalid temperature value."""
    with pytest.raises(ValueError, match="temperature must be between 0.0 and 2.0"):
        ChatMetaLlama(
            model_name="test-model",
            temperature=2.5,  # Above the max allowed
            client=MagicMock(spec=LlamaAPIClient),
        )


def test_ai_message_non_string_content():
    """Test AIMessage with non-string content gets converted to string."""
    # Create AIMessage with non-string content
    with patch("warnings.warn") as mock_warn:
        # Using a list content will trigger the warning path when converted
        ai_message = AIMessage(content=["item1", "item2"])
        result = _lc_message_to_llama_message_param(ai_message)

    # Verify the result has the stringified content
    assert result["role"] == "assistant"
    assert result["content"] == "['item1', 'item2']"
    assert mock_warn.called


def test_tool_call_string_conversion():
    """Test tool call gets converted appropriately."""
    # Create a valid ToolMessage with a string that needs no conversion
    message = ToolMessage(content="test result", tool_call_id="123")

    # Call the function
    result = _lc_message_to_llama_message_param(message)

    # Verify the result
    assert result["role"] == "tool"
    assert result["content"] == "test result"
    assert result["tool_call_id"] == "123"


def test_non_standard_tool_call_handling():
    """Test handling of non-standard tool calls using our own mock objects."""

    # Create a class that represents a ToolCall-like object but doesn't have all
    # the required attributes of a standard ToolCall
    class CustomToolCall:
        def __init__(self):
            self.name = "custom_tool"
            # Missing id attribute
            # Missing args attribute

    # Create an AIMessage with our custom tool call
    message = AIMessage(content="test")
    # Manually set tool_calls after creation to bypass validation
    message.tool_calls = [CustomToolCall()]

    # Patch out the logger warning that would be triggered
    with patch("warnings.warn") as mock_warn:
        # Call the function and verify it handles the non-standard object
        try:
            result = _lc_message_to_llama_message_param(message)
            # If it gets here without error, the test passes
            assert "tool_calls" in result
        except Exception as e:
            # The function should handle or wrap exceptions from non-standard objects
            pytest.fail(f"Function should handle non-standard tool calls: {e}")

        # We should have seen a warning
        assert mock_warn.called
