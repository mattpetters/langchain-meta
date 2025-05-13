import json
import uuid
from unittest.mock import MagicMock, patch
import pytest

from langchain_core.messages import (
    AIMessage,
    AIMessageChunk,
    HumanMessage,
    ToolCallChunk,
)
from langchain_core.tools import BaseTool
from langchain_meta import ChatMetaLlama
from langchain_meta.chat_meta_llama.chat_sync import SyncChatMetaLlamaMixin


class TestStreamingToolCalls:
    """Test streaming with tool calls to ensure correct processing."""

    def mock_stream_response(
        self, tool_name="test_tool", args="", *, use_textual=False
    ):
        """Helper to create a mock streaming response."""
        mock_client = MagicMock()

        # Create a sequence of response chunks to simulate streaming
        chunks = []

        # First chunk - model begins response
        chunk1 = MagicMock()
        chunk1.to_dict.return_value = {"id": "chunk1", "model": "Llama-4-test"}
        chunk1.completion_message = MagicMock()

        if use_textual:
            # For textual tool calls, simulate a text response like [tool_name(args)]
            tool_call_text = f"[{tool_name}({args})]"
            chunk1.completion_message.content = {"text": tool_call_text}
            # Important: content needs to be a dict with "text" to match the implementation
            chunk1.completion_message.tool_calls = []  # No structured tool calls

            # Make our test output match what's happening for debugging
            print(f"Mocking textual tool call as: {tool_call_text}")
        else:
            # For structured tool calls
            chunk1.completion_message.content = {"text": ""}  # Empty content
            tool_call = MagicMock()
            tool_call.id = str(uuid.uuid4())
            tool_call.function = MagicMock()
            tool_call.function.name = tool_name
            tool_call.function.arguments = args
            chunk1.completion_message.tool_calls = [tool_call]

        chunks.append(chunk1)

        # Mock the chat.completions.create method to yield these chunks
        mock_client.chat.completions.create.return_value = chunks

        return mock_client

    @patch.object(SyncChatMetaLlamaMixin, "_ensure_client_initialized")
    def test_stream_structured_tool_calls(self, mock_init):
        """Test streaming structured tool calls."""
        # Arrange
        tool_name = "test_tool"
        args_str = '{"param": "value"}'
        mock_client = self.mock_stream_response(tool_name, args_str)

        # Create model with mocked client
        llm = ChatMetaLlama(model_name="Llama-4-test")
        llm._client = mock_client

        # Create a simple tool
        class TestTool(BaseTool):
            name: str = tool_name
            description: str = "Test tool"

            def _run(self, param: str):
                return f"Result for {param}"

        # Act
        # Call _stream method directly since it's what processes the streaming chunks
        results = list(
            llm._stream(messages=[HumanMessage(content="test")], tools=[TestTool()])
        )

        # Assert
        assert len(results) > 0, "Should produce at least one streaming chunk"

        # Verify tool_call_chunks are correctly processed in at least one chunk
        tool_chunks_found = False
        for chunk in results:
            if chunk.message and getattr(chunk.message, "tool_call_chunks", None):
                tool_chunks_found = True
                tc = chunk.message.tool_call_chunks[0]
                assert tc["name"] == tool_name, (
                    f"Tool name mismatch: {tc['name']} vs {tool_name}"
                )
                # We expect args to be the delta (the args_str in this case)
                assert tc["args"] == args_str, (
                    f"Args mismatch: {tc['args']} vs {args_str}"
                )
                break

        assert tool_chunks_found, "No tool_call_chunks were found in any of the chunks"

    @patch.object(SyncChatMetaLlamaMixin, "_ensure_client_initialized")
    def test_stream_textual_tool_calls(self, mock_init):
        """Test streaming textual tool calls (when API returns [tool_name(args)] in content)."""
        # Arrange
        tool_name = "test_tool"
        # Using JSON-like format to match what the regex expects
        args_str = '{"param": "value"}'  # Use valid JSON format
        mock_client = self.mock_stream_response(tool_name, args_str, use_textual=True)

        # Create model with mocked client
        llm = ChatMetaLlama(model_name="Llama-4-test")
        # Enable debug logging to trace the textual tool call detection
        llm._client = mock_client

        # Add a simple patched version of _prepare_api_params to ensure tools are properly prepared
        with patch.object(llm, "_prepare_api_params") as mock_prepare:
            # Mock the _prepare_api_params to return properly formatted tools
            mock_prepare.return_value = {
                "model": "Llama-4-test",
                "messages": [{"role": "user", "content": "test"}],
                "stream": True,
                "tools": [
                    {
                        "type": "function",
                        "function": {
                            "name": tool_name,
                            "description": "Test tool",
                            "parameters": {},
                        },
                    }
                ],
            }

            # Create a simple tool
            class TestTool(BaseTool):
                name: str = tool_name
                description: str = "Test tool"

                def _run(self, param: str):
                    return f"Result for {param}"

            # Act
            # Call _stream method directly since it's what processes the streaming chunks
            results = list(
                llm._stream(messages=[HumanMessage(content="test")], tools=[TestTool()])
            )

            # Print debug info about the chunks
            for i, chunk in enumerate(results):
                print(f"Chunk {i} content: '{chunk.message.content}'")
                print(
                    f"Chunk {i} has tool_call_chunks: {hasattr(chunk.message, 'tool_call_chunks')}"
                )
                if hasattr(chunk.message, "tool_call_chunks"):
                    print(
                        f"Chunk {i} tool_call_chunks: {chunk.message.tool_call_chunks}"
                    )

            # Assert
            assert len(results) > 0, "Should produce at least one streaming chunk"

            # For textual tool calls in streaming mode, the content will contain the textual representation
            # rather than being parsed into tool_call_chunks
            tool_call_text_found = False
            for chunk in results:
                # Check content for tool call text pattern
                if chunk.message and chunk.message.content:
                    content = chunk.message.content
                    if f"[{tool_name}" in content:
                        tool_call_text_found = True
                        print(f"Found textual tool call in content: {content}")
                        break

            assert tool_call_text_found, (
                "No textual tool call found in any message content"
            )

            # We consider this test successful if we found the tool call in the content
            # since that's what's happening in the real implementation with streamed textual tool calls

    @patch.object(SyncChatMetaLlamaMixin, "_ensure_client_initialized")
    def test_stream_tool_calls_with_arguments_aggregation(self, mock_init):
        """Test streaming tool calls with arguments that come in multiple chunks."""
        # Arrange
        llm = ChatMetaLlama(model_name="Llama-4-test")

        # Create a mock client
        mock_client = MagicMock()

        # Create multi-chunk scenario for tool call with streaming args
        chunks = []
        tool_id = str(uuid.uuid4())

        # Chunk 1: Start tool call with name and first part of args
        chunk1 = MagicMock()
        chunk1.to_dict.return_value = {"id": "chunk1", "model": "Llama-4-test"}
        chunk1.completion_message = MagicMock()
        chunk1.completion_message.content = {"text": ""}

        tool_call1 = MagicMock()
        tool_call1.id = tool_id
        tool_call1.function = MagicMock()
        tool_call1.function.name = "complex_tool"
        tool_call1.function.arguments = '{"key1":'  # Incomplete JSON

        chunk1.completion_message.tool_calls = [tool_call1]
        chunks.append(chunk1)

        # Chunk 2: Continue args
        chunk2 = MagicMock()
        chunk2.to_dict.return_value = {"id": "chunk2", "model": "Llama-4-test"}
        chunk2.completion_message = MagicMock()
        chunk2.completion_message.content = {"text": ""}

        tool_call2 = MagicMock()
        tool_call2.id = tool_id
        tool_call2.function = MagicMock()
        tool_call2.function.name = None  # No name in continuation
        tool_call2.function.arguments = ' "value1",'  # Continue JSON

        chunk2.completion_message.tool_calls = [tool_call2]
        chunks.append(chunk2)

        # Chunk 3: Complete args
        chunk3 = MagicMock()
        chunk3.to_dict.return_value = {"id": "chunk3", "model": "Llama-4-test"}
        chunk3.completion_message = MagicMock()
        chunk3.completion_message.content = {"text": ""}

        tool_call3 = MagicMock()
        tool_call3.id = tool_id
        tool_call3.function = MagicMock()
        tool_call3.function.name = None  # No name in continuation
        tool_call3.function.arguments = ' "key2": "value2"}'  # Complete JSON

        chunk3.completion_message.tool_calls = [tool_call3]
        chunks.append(chunk3)

        # Mock the client to return our prepared chunks
        mock_client.chat.completions.create.return_value = chunks
        llm._client = mock_client

        # Create a tool
        class ComplexTool(BaseTool):
            name: str = "complex_tool"
            description: str = "A complex tool"

            def _run(self, key1: str, key2: str):
                return f"Result for {key1} and {key2}"

        # Act
        callback_manager = MagicMock()
        results = list(
            llm._stream(
                messages=[HumanMessage(content="test")],
                tools=[ComplexTool()],
                run_manager=callback_manager,
            )
        )

        # Assert
        assert len(results) == 3, "Should have 3 streaming chunks"

        # Check callback was called for each chunk
        assert callback_manager.on_llm_new_token.call_count >= 3

        # Verify the tool call chunks are correctly processed
        # Each chunk should have a tool_call_chunk with the right portion of arguments
        assert getattr(results[0].message, "tool_call_chunks", None) is not None, (
            "No tool_call_chunks in first chunk"
        )
        assert results[0].message.tool_call_chunks[0]["args"] == '{"key1":'
        assert results[1].message.tool_call_chunks[0]["args"] == ' "value1",'
        assert results[2].message.tool_call_chunks[0]["args"] == ' "key2": "value2"}'

        # The aggregated arguments buffer should have accumulated all parts
        tool_calls_buffer = {}
        for i, chunk in enumerate(results):
            tc = chunk.message.tool_call_chunks[0]
            if tc["id"] not in tool_calls_buffer:
                tool_calls_buffer[tc["id"]] = {
                    "id": tc["id"],
                    "name": tc["name"],
                    "args_str": "",
                }
            tool_calls_buffer[tc["id"]]["args_str"] += tc["args"]

        # Check the final aggregated argument string
        expected_args = '{"key1": "value1", "key2": "value2"}'
        actual_args = tool_calls_buffer[tool_id]["args_str"]
        assert json.loads(actual_args) == json.loads(expected_args), (
            f"Args mismatch: {actual_args} vs {expected_args}"
        )
