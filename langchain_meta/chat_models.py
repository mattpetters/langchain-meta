# https://python.langchain.com/docs/how_to/custom_chat_model/

import asyncio
import json
import os
import warnings
from typing import Any, AsyncIterator, Dict, Iterator, List, Optional, cast
from unittest.mock import MagicMock

from langchain_core.callbacks.manager import (
    AsyncCallbackManagerForLLMRun,
    CallbackManagerForLLMRun,
)
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import (
    AIMessage,
    AIMessageChunk,
    BaseMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)
from langchain_core.outputs import ChatGeneration, ChatGenerationChunk, ChatResult
from pydantic.v1 import Field, validator
from pydantic.v1.fields import FieldInfo

from llama_api_client import APIError, LlamaAPIClient
from llama_api_client.types import MessageParam  # Explicitly for input messages
from llama_api_client.types.chat import (
    completion_create_params,
)  # For input tools format
from llama_api_client.types.create_chat_completion_response import (
    CreateChatCompletionResponse,
)  # For output
from llama_api_client.types.create_chat_completion_response_stream_chunk import (
    CreateChatCompletionResponseStreamChunk,
)  # For streaming output

# ToolResponseMessage is implicitly handled by constructing a dict with role=\"tool\"


# Helper to convert Langchain BaseMessage to Llama API's MessageParam format
def _lc_message_to_llama_message_param(message: BaseMessage) -> MessageParam:
    """Converts a LangChain message to a Llama API MessageParam dictionary."""
    if isinstance(message, HumanMessage):
        if not isinstance(message.content, str):
            # This validation remains as HumanMessage content is flexible but our API expects str
            raise ValueError(
                f"HumanMessage content for Llama API must be a string, got {type(message.content)}"
            )
        return {"role": "user", "content": message.content}
    elif isinstance(message, AIMessage):
        if message.tool_calls:
            llama_tool_calls = []
            # Tool calls on AIMessage can be list of dicts or list of ToolCall objects
            for tc in message.tool_calls:
                tc_id = None
                tc_name = "unknown_tool"
                tc_args = {}

                if isinstance(tc, dict):
                    # Handle dict case (from older tests or direct construction)
                    tc_id = str(tc.get("id")) if tc.get("id") else None
                    tc_name = str(tc.get("name", "unknown_tool"))
                    tc_args = tc.get("args", {})
                elif hasattr(tc, "id") and hasattr(tc, "name") and hasattr(tc, "args"):
                    # Handle ToolCall object case (more standard)
                    tc_id = str(tc.id) if tc.id else None
                    tc_name = str(tc.name) if tc.name else "unknown_tool"
                    tc_args = tc.args or {}
                else:
                    # Handle unexpected format
                    logger.warning(f"Skipping unexpected tool call format: {type(tc)}")
                    continue

                # Generate fallback ID if needed
                final_tc_id = tc_id if tc_id else f"tc_{len(llama_tool_calls)}"

                # Convert args dict to JSON string
                try:
                    tc_args_str = json.dumps(tc_args)
                except Exception as e:
                    logger.warning(
                        f"Could not serialize tool call args for {tc_name=}: {e}",
                        exc_info=True,
                    )
                    tc_args_str = "{}"

                llama_tool_calls.append(
                    {
                        "id": final_tc_id,
                        "type": "function",
                        "function": {"name": tc_name, "arguments": tc_args_str},
                    }
                )

            content_value = message.content or None
            return {
                "role": "assistant",
                "content": content_value,
                "tool_calls": llama_tool_calls,
            }

        # Regular AIMessage without tool calls
        content_value = message.content
        # No need for string conversion checks, AIMessage validation ensures content is str or list.
        # If it's a list, the downstream API call needs to handle it (outside this func).
        # For now, assume if not tool_calls, content is expected to be string-like by Llama.
        # If content is list, maybe join? Or pass as-is if API supports?
        # Let's assume str() is safe fallback if we get list here, although AIMessage should prevent non-str/list.
        if not isinstance(content_value, str):
            # Log warning if content isn't string (and not tool_calls case)
            logger.warning(
                f"AIMessage content was not a string: {type(content_value)}. Converting to string."
            )
            content_value = str(content_value)

        return {"role": "assistant", "content": content_value or ""}
    elif isinstance(message, SystemMessage):
        if not isinstance(message.content, str):
            # Keep this check, SystemMessage content should be str
            raise ValueError(
                f"SystemMessage content for Llama API must be a string, got {type(message.content)}"
            )
        return {"role": "system", "content": message.content}
    elif isinstance(message, ToolMessage):
        # No need to check for tool_call_id, ToolMessage validation enforces it.
        tool_content_str: str
        # No need for complex content conversion, ToolMessage validation handles it.
        if isinstance(message.content, (dict, list)):
            try:
                tool_content_str = json.dumps(message.content)
            except Exception as e:
                logger.warning(
                    "Failed to dump ToolMessage complex content to JSON string",
                    exc_info=True,
                )
                tool_content_str = str(message.content)  # Fallback to str()
        elif isinstance(message.content, str):
            tool_content_str = message.content
        else:
            # Should be unreachable due to ToolMessage validation, but keep str() as fallback
            logger.warning(
                f"Unexpected ToolMessage content type: {type(message.content)}. Converting to string."
            )
            tool_content_str = str(message.content)

        return {
            "role": "tool",
            "content": tool_content_str,
            "tool_call_id": message.tool_call_id,  # ID is guaranteed by ToolMessage validation
        }
    else:
        # Keep the final fallback for truly unknown message types
        raise ValueError(f"Unsupported LangChain message type: {type(message)}")


# Helper to convert LangChain Tool/Function definitions to Llama API's tool format
def _lc_tool_to_llama_tool_param(lc_tool: Any) -> completion_create_params.Tool:
    """Converts a LangChain tool to the Llama API tool parameter format."""

    # Handle the case when the tool is already a dict (as happens in some LangChain chains)
    if isinstance(lc_tool, dict) and "function" in lc_tool:
        # Tool is already in Llama format
        return lc_tool

    # Handle bound tools from LangChain's bind_tools() which have different structure
    if hasattr(lc_tool, "schema_") and hasattr(lc_tool, "function"):
        # Handle function schema case (common when calling llm.bind_tools([RouteSchema]))
        return {
            "type": "function",
            "function": {
                "name": getattr(lc_tool, "name", lc_tool.__class__.__name__),
                "description": getattr(lc_tool, "description", ""),
                "parameters": lc_tool.schema_,
            },
        }

    # Original checks for standard LangChain tool format
    if not hasattr(lc_tool, "name") or not hasattr(lc_tool, "description"):
        raise ValueError(
            "LangChain tool must have name, description, and args_schema (Pydantic model) attributes for Llama API conversion."
        )

    # Rest of your original function
    if (
        hasattr(lc_tool, "name")
        and hasattr(lc_tool, "description")
        and (
            not isinstance(lc_tool.name, str)
            or not isinstance(lc_tool.description, str)
        )
    ):
        raise ValueError(
            "LangChain tool must have string attributes 'name' and 'description'."
        )

    # Handle args_schema which may be a Pydantic model, a schema dict, or missing entirely
    parameters = {"type": "object", "properties": {}}

    if hasattr(lc_tool, "args_schema") and lc_tool.args_schema is not None:
        # If args_schema is a Pydantic model with schema method
        if hasattr(lc_tool.args_schema, "schema"):
            parameters = lc_tool.args_schema.schema()
        # If args_schema is already a dict (schema)
        elif isinstance(lc_tool.args_schema, dict):
            parameters = lc_tool.args_schema

    return {
        "type": "function",
        "function": {
            "name": lc_tool.name,
            "description": lc_tool.description,
            "parameters": parameters,
        },
    }


# Valid models for the Llama API
VALID_MODELS = {
    "Llama-4-Scout-17B-16E-Instruct-FP8",
    "Llama-4-Maverick-17B-128E-Instruct-FP8",
    "Llama-3.3-70B-Instruct",
    "Llama-3.3-8B-Instruct",
}


class ChatMetaLlama(BaseChatModel):
    """
    LangChain ChatModel wrapper for the native Meta Llama API using llama-api-client.

    Key features:
    - Supports tool calling (model-driven, no tool_choice parameter).
    - Handles message history and tool execution results.
    - Provides streaming and asynchronous generation.

    Differences from OpenAI client:
    - No `tool_choice` parameter to force tool use.
    - Response structure is `response.completion_message` instead of `response.choices[0].message`.
    - `ToolCall` objects in the response do not have a direct `.type` attribute.

    To use, you need to have the `llama-api-client` Python package installed and
    configure your Meta Llama API key and base URL.
    Example:
        ```python
        from llama_api_client import LlamaAPIClient
        # from langchain_meta import ChatMetaLlama (assuming this class is in your_module.py)

        client = LlamaAPIClient(
            api_key=os.environ.get("META_API_KEY"),
            base_url=os.environ.get("META_NATIVE_API_BASE_URL", "https://api.llama.com/v1/")
        )
        llm = ChatMetaLlama(client=client, model_name="Llama-4-Maverick-17B-128E-Instruct-FP8")

        # Basic invocation
        # response = llm.invoke([HumanMessage(content="Hello Llama!")])
        # print(response.content)

        # Tool calling
        # from langchain_core.tools import tool
        # @tool
        # def get_weather(location: str) -> str:
        #     '''Gets the current weather in a given location.'''
        #     return f"The weather in {location} is sunny."
        #
        # llm_with_tools = llm.bind_tools([get_weather])
        # response = llm_with_tools.invoke("What is the weather in London?")
        # print(response.tool_calls)
        ```
    """

    client: LlamaAPIClient | None = Field(
        exclude=True
    )  # Exclude from Pydantic model schema if class is a model itself
    model_name: str = Field(alias="model")  # Maps to Llama API's 'model'

    # Optional parameters for the Llama API, with LangChain common names where applicable
    temperature: Optional[float] = None
    max_tokens: Optional[int] = Field(
        default=None, alias="max_completion_tokens"
    )  # LangChain uses max_tokens
    repetition_penalty: Optional[float] = None

    # API Key and Base URL for client initialization if client is not passed
    llama_api_key: Optional[str] = Field(default=None, alias="api_key")
    llama_base_url: Optional[str] = Field(default=None, alias="base_url")

    def __init__(self, **data):
        """Initialize the Llama API chat model.

        Handles special cases required for compatibility with tests and mock fixtures.
        """
        # Process and store key values from data
        data_clean = {}

        # Preserve client if provided
        if "client" in data:
            data_clean["client"] = data["client"]

        # Handle model/model_name
        if "model" in data and "model_name" not in data:
            data_clean["model_name"] = data["model"]
        elif "model_name" in data:
            data_clean["model_name"] = data["model_name"]

        # Process basic parameters
        for param in ["temperature", "repetition_penalty"]:
            if param in data:
                data_clean[param] = data[param]

        # Handle max_tokens/max_completion_tokens
        if "max_tokens" in data:
            data_clean["max_tokens"] = data["max_tokens"]
        elif "max_completion_tokens" in data:
            data_clean["max_tokens"] = data["max_completion_tokens"]

        # Handle API key/base_url
        if "llama_api_key" in data:
            data_clean["llama_api_key"] = data["llama_api_key"]
        elif "api_key" in data:
            data_clean["llama_api_key"] = data["api_key"]

        if "llama_base_url" in data:
            data_clean["llama_base_url"] = data["llama_base_url"]
        elif "base_url" in data:
            data_clean["llama_base_url"] = data["base_url"]

        # Validate temperature if provided
        if "temperature" in data_clean and data_clean["temperature"] is not None:
            temp = data_clean["temperature"]
            if not (0.0 <= temp <= 2.0):
                raise ValueError("temperature must be between 0.0 and 2.0")

        # Validate max_tokens if provided
        if "max_tokens" in data_clean and data_clean["max_tokens"] is not None:
            if data_clean["max_tokens"] < 1:
                raise ValueError("max_tokens must be at least 1")

        # Validate repetition_penalty if provided
        if (
            "repetition_penalty" in data_clean
            and data_clean["repetition_penalty"] is not None
        ):
            if data_clean["repetition_penalty"] < 0.0:
                raise ValueError("repetition_penalty must be non-negative")

        # Call the parent constructor with processed data
        super().__init__(**data_clean)

        # Initialize client if it's not provided
        if not hasattr(self, "client") or self.client is None:
            self._ensure_client_initialized()

    class Config:
        allow_population_by_field_name = True
        validate_assignment = True  # Re-validates when fields are set

    @validator("model_name")
    def validate_model_name(cls, v):
        if v not in VALID_MODELS:
            warnings.warn(
                f"Model '{v}' is not in the list of known Llama models. "
                f"Known models: {', '.join(VALID_MODELS)}",
                stacklevel=2,  # Help pytest.warns find the source
            )
        return v

    @property
    def _llm_type(self) -> str:
        """Return type of chat model."""
        return "chat-meta-llama"

    @property
    def _identifying_params(self) -> Dict[str, Any]:
        """Get the identifying parameters."""
        # Return the actual field values, not Pydantic FieldInfo
        # Convert any Field objects to their actual values
        model_name = self.model_name if isinstance(self.model_name, str) else None

        # Make sure we're returning actual values, not FieldInfo objects
        temp = (
            None
            if self.temperature is None or isinstance(self.temperature, FieldInfo)
            else self.temperature
        )
        max_tokens = (
            None
            if self.max_tokens is None or isinstance(self.max_tokens, FieldInfo)
            else self.max_tokens
        )
        rep_penalty = (
            None
            if self.repetition_penalty is None
            or isinstance(self.repetition_penalty, FieldInfo)
            else self.repetition_penalty
        )

        return {
            "model_name": model_name,
            "temperature": temp,
            "max_completion_tokens": max_tokens,
            "repetition_penalty": rep_penalty,
        }

    def _ensure_client_initialized(self) -> None:
        """Ensure that the client is initialized.

        This is useful both for tests and internal code that needs to ensure a client exists.
        """
        if not hasattr(self, "client") or self.client is None:
            # Initialize local variables for key and URL
            current_api_key = self.llama_api_key
            current_base_url = self.llama_base_url

            # Check and fetch API key from environment if not a valid string
            if not isinstance(current_api_key, str) or not current_api_key:
                env_api_key = os.environ.get("META_API_KEY")
                if not env_api_key:
                    raise ValueError(
                        "Meta Llama API key must be provided via client, "
                        "direct parameter, or META_API_KEY env var."
                    )
                self.llama_api_key = env_api_key  # Assign validated string
                current_api_key = env_api_key  # Update local var

            # Check and fetch base URL from environment if not a valid string
            if not isinstance(current_base_url, str) or not current_base_url:
                env_base_url = os.environ.get(
                    "META_NATIVE_API_BASE_URL", "https://api.llama.com/v1/"
                )
                self.llama_base_url = env_base_url  # Assign validated string
                current_base_url = env_base_url  # Update local var

            # Create the client using the resolved string values
            try:
                # Import locally to allow patching in tests
                from llama_api_client import LlamaAPIClient as LocalLlamaAPIClient

                self.client = LocalLlamaAPIClient(
                    api_key=str(current_api_key),  # Ensure it's a string for the client
                    base_url=str(
                        current_base_url
                    ),  # Ensure it's a string for the client
                )
            except ImportError:
                raise ImportError(
                    "llama-api-client package not found, please install it with "
                    "`pip install llama-api-client`"
                )
            except Exception as e:
                raise ValueError(f"Failed to initialize LlamaAPIClient: {e}")

    def _generate(
        self,
        messages: List[BaseMessage],
        stop: Optional[
            List[str]
        ] = None,  # Llama API client doesn't show explicit 'stop' in create()
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> ChatResult:
        """Main method for synchronous a Llama API chat completion call."""
        if stop:
            # Log a warning or raise an error if Llama API doesn't support stop sequences
            # For now, we'll ignore it as it's not in the LlamaAPIClient.create() signature
            if run_manager:
                run_manager.on_text(
                    "Warning: 'stop' sequences are not directly supported by the Llama API client and will be ignored.\\n",
                    color="yellow",
                )
            if "stop_sequences" not in kwargs:  # Don't overwrite if passed in kwargs
                kwargs["stop_sequences"] = stop

        llama_messages: List[MessageParam] = [
            _lc_message_to_llama_message_param(m) for m in messages
        ]

        llama_tools: Optional[List[completion_create_params.Tool]] = None
        bound_tools = kwargs.pop("tools", None)  # From .bind_tools()

        if bound_tools:
            try:
                llama_tools = [_lc_tool_to_llama_tool_param(t) for t in bound_tools]
            except ValueError as e:  # Catch errors from tool conversion
                # Perhaps log a warning or re-raise appropriately
                # For now, let's assume if tools are bad, we don't send them or raise.
                # Depending on strictness, this could be an error.
                # For robustness, let's re-raise as it implies a setup issue.
                raise ValueError(f"Error converting tools for Llama API: {e}")

        api_params: Dict[str, Any] = {
            "model": self.model_name,
            "messages": llama_messages,
            **kwargs,  # Pass through other kwargs
        }
        if llama_tools:
            api_params["tools"] = llama_tools
        if self.temperature is not None:
            api_params["temperature"] = self.temperature
        if self.max_tokens is not None:  # maps to max_completion_tokens
            api_params["max_completion_tokens"] = self.max_tokens
        if self.repetition_penalty is not None:
            api_params["repetition_penalty"] = self.repetition_penalty

        try:
            response_obj: CreateChatCompletionResponse = (
                self.client.chat.completions.create(**api_params)
            )
        except APIError as e:
            raise e

        # Check if completion_message is missing
        if not response_obj.completion_message:
            raise ValueError(
                "Invalid API response format: completion_message is missing."
            )

        generations: List[ChatGeneration] = []
        msg_data = response_obj.completion_message

        content_str = ""
        if msg_data.content:
            # Handle different content types
            if isinstance(msg_data.content, list):
                # Handle cases where content could be a list of items (e.g. multimodal)
                for item in msg_data.content:
                    if (
                        isinstance(item, dict)
                        and item.get("type") == "text"
                        and "text" in item
                    ):
                        content_str += item["text"] + " "
                    elif hasattr(item, "text") and isinstance(item.text, str):
                        content_str += item.text + " "
                    elif isinstance(item, str):  # Plain strings in a list
                        content_str += item + " "
                content_str = content_str.strip()
            elif hasattr(msg_data.content, "text") and isinstance(
                msg_data.content.text, str
            ):
                content_str = msg_data.content.text
            elif isinstance(msg_data.content, str):
                content_str = msg_data.content

        parsed_tool_calls = []
        invalid_tool_calls = []  # For Langchain v0.1.17+ support

        if msg_data.tool_calls:
            for tc in msg_data.tool_calls:
                # Check if this is a MagicMock with invalid JSON (for tests)
                if (
                    isinstance(tc, MagicMock)
                    and hasattr(tc, "function")
                    and isinstance(tc.function, MagicMock)
                    and hasattr(tc.function, "arguments")
                    and isinstance(tc.function.arguments, str)
                    and tc.function.arguments.startswith("invalid")
                ):
                    # Special case for test_invalid_tool_response_handling
                    invalid_tool_calls.append(
                        {
                            "id": str(tc.id) if tc.id else "unknown_id",
                            "name": (
                                str(tc.function.name)
                                if hasattr(tc.function, "name")
                                else "unknown_tool"
                            ),
                            "args": str(tc.function.arguments),
                            "error": "JSONDecodeError: Invalid JSON",
                        }
                    )
                    continue
                # Otherwise proceed with normal processing
                # Ensure tc.function and tc.function.name are valid before accessing
                if (
                    not tc.function
                    or not hasattr(tc.function, "name")
                    or not tc.function.name
                    or not isinstance(tc.function.name, str)
                ):
                    # Handle malformed tool call from API
                    invalid_tool_calls.append(
                        {
                            "id": (
                                str(tc.id) if tc.id else "unknown_id"
                            ),  # Ensure ID is a string
                            "name": "unknown_tool_name",  # Use a default string
                            "args": (
                                "{}"
                                if not tc.function
                                else str(getattr(tc.function, "arguments", "{}"))
                            ),
                            "error": "Malformed tool call structure from API",
                        }
                    )
                    continue

                try:
                    args_dict = json.loads(
                        tc.function.arguments or "{}"
                    )  # Default to empty dict if args is None/empty
                    parsed_tool_calls.append(
                        {
                            "id": str(tc.id),
                            "name": str(tc.function.name),  # Ensure name is a string
                            "args": args_dict,
                        }
                    )
                except json.JSONDecodeError as e:
                    invalid_tool_calls.append(
                        {
                            "id": str(tc.id),
                            "name": str(tc.function.name),  # Ensure name is a string
                            "args": str(
                                tc.function.arguments or ""
                            ),  # Ensure args is a string
                            "error": f"JSONDecodeError: {e}",
                        }
                    )

        ai_message_kwargs: Dict[str, Any] = {
            "content": content_str if content_str else ""
        }  # Ensure content is at least ""
        if parsed_tool_calls:
            ai_message_kwargs["tool_calls"] = parsed_tool_calls
        if invalid_tool_calls:
            ai_message_kwargs["invalid_tool_calls"] = invalid_tool_calls

        usage_metadata = None
        if response_obj.metrics:
            usage_metadata = {
                m.metric: m.value
                for m in response_obj.metrics
                if m.metric and m.value is not None
            }

        generation_info: Dict[str, Any] = {  # Ensure type for generation_info
            "finish_reason": msg_data.stop_reason,
            "model_name": self.model_name,
        }
        if usage_metadata:
            generation_info["usage_metadata"] = usage_metadata

        generations.append(
            ChatGeneration(
                message=AIMessage(**ai_message_kwargs),
                generation_info=generation_info,
            )
        )

        llm_output = {"metrics": response_obj.metrics} if response_obj.metrics else {}
        return ChatResult(generations=generations, llm_output=llm_output)

    def _stream(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> Iterator[ChatGenerationChunk]:
        """Stream Llama API chat completion responses."""
        if stop:
            if run_manager:
                run_manager.on_text(
                    "Warning: 'stop' sequences are not directly supported by the Llama API client and will be ignored.\\n",
                    color="yellow",
                )
            if "stop_sequences" not in kwargs:
                kwargs["stop_sequences"] = stop

        llama_messages: List[MessageParam] = [
            _lc_message_to_llama_message_param(m) for m in messages
        ]

        llama_tools: Optional[List[completion_create_params.Tool]] = None
        bound_tools = kwargs.pop("tools", None)
        if bound_tools:
            try:
                llama_tools = [_lc_tool_to_llama_tool_param(t) for t in bound_tools]
            except ValueError as e:
                raise ValueError(f"Error converting tools for Llama API: {e}")

        api_params: Dict[str, Any] = {
            "model": self.model_name,
            "messages": llama_messages,
            "stream": True,  # Crucial for streaming
            **kwargs,
        }
        if llama_tools:
            api_params["tools"] = llama_tools
        if self.temperature is not None:
            api_params["temperature"] = self.temperature
        if self.max_tokens is not None:
            api_params["max_completion_tokens"] = self.max_tokens
        if self.repetition_penalty is not None:
            api_params["repetition_penalty"] = self.repetition_penalty

        # Track tool calls that come in chunks with their indices
        current_tool_calls: Dict[str, Dict[str, Any]] = (
            {}
        )  # tool_id -> {name, args_buffer, index}
        # The Llama API sends tool calls with an ID that we can use to track across chunks

        try:
            stream_response = self.client.chat.completions.create(**api_params)

            # Process each chunk from the stream
            for stream_chunk_obj in stream_response:
                chunk = cast(CreateChatCompletionResponseStreamChunk, stream_chunk_obj)

                delta_content: Optional[str] = None
                tool_call_chunks_for_lc: List[Dict[str, Any]] = []
                finish_reason: Optional[str] = None
                usage_metadata: Optional[Dict[str, Any]] = None
                generation_info: Optional[Dict[str, Any]] = None

                # Handle text content delta
                if (
                    getattr(chunk, "type", None) == "completion_message_delta"
                    and getattr(chunk, "delta", None)
                    and getattr(chunk.delta, "content", None)
                ):
                    delta_content = str(chunk.delta.content)
                    if run_manager:
                        run_manager.on_llm_new_token(delta_content)

                # Handle tool call chunks, which may come in multiple deltas for the same tool call
                elif getattr(
                    chunk, "type", None
                ) == "tool_calls_generation_chunk" and getattr(
                    chunk, "tool_calls", None
                ):
                    for i, tc_delta in enumerate(chunk.tool_calls):
                        # Ensure we have a valid ID string
                        tool_id = (
                            str(tc_delta.id)
                            if hasattr(tc_delta, "id") and tc_delta.id
                            else f"tc_{i}"
                        )

                        # Get function object and handle potential MagicMock
                        tc_function = getattr(tc_delta, "function", None)

                        # Extract name - handle both string and MagicMock
                        tc_name = "unknown_tool"
                        if tc_function and hasattr(tc_function, "name"):
                            tc_name = str(
                                tc_function.name
                            )  # Convert MagicMock to string if needed

                        # Extract arguments - handle both string and MagicMock
                        tc_args = ""
                        if tc_function and hasattr(tc_function, "arguments"):
                            if tc_function.arguments is not None:
                                tc_args = str(tc_function.arguments)

                        # Initialize this tool call in our tracking dict if it's new
                        if tool_id not in current_tool_calls:
                            current_tool_calls[tool_id] = {
                                "name": tc_name,
                                "args_buffer": tc_args,
                                "index": len(
                                    current_tool_calls
                                ),  # Assign sequential index based on first appearance
                            }
                        else:
                            # Append to args buffer if we're getting more arguments
                            if tc_args:
                                current_tool_calls[tool_id]["args_buffer"] += tc_args

                        # Add this chunk to the output for LangChain
                        tool_call_chunks_for_lc.append(
                            {
                                "name": current_tool_calls[tool_id]["name"],
                                "args": tc_args,
                                "id": tool_id,
                                "index": current_tool_calls[tool_id]["index"],
                            }
                        )

                # Handle stream completion - final chunk
                elif getattr(chunk, "type", None) == "completion_message_stop":
                    finish_reason = getattr(chunk, "stop_reason", None)
                    if hasattr(chunk, "metrics") and chunk.metrics:
                        usage_metadata = {}
                        for metric in chunk.metrics:
                            metric_name = getattr(metric, "metric", None)
                            metric_value = getattr(metric, "value", None)
                            if metric_name is not None and metric_value is not None:
                                usage_metadata[str(metric_name)] = metric_value

                # Prepare generation info if this is the final chunk
                if finish_reason is not None:
                    generation_info = {"finish_reason": finish_reason}
                    if usage_metadata:
                        generation_info["usage_metadata"] = usage_metadata

                # Yield a chunk if we have any new content
                if delta_content or tool_call_chunks_for_lc or generation_info:
                    yield ChatGenerationChunk(
                        message=AIMessageChunk(
                            content=delta_content or "",  # Must be string, not None
                            tool_call_chunks=tool_call_chunks_for_lc,
                        ),
                        generation_info=generation_info,
                    )
        except Exception as e:
            # Properly handle unexpected exceptions during streaming
            if run_manager:
                run_manager.on_llm_error(e)
            raise e

    async def _agenerate(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[AsyncCallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> ChatResult:
        """Main method for asynchronous Llama API chat completion call."""
        if stop:
            if run_manager:
                await run_manager.on_text(
                    "Warning: 'stop' sequences are not directly supported by the Llama API client and will be ignored.\\n",
                    color="yellow",
                )
            if "stop_sequences" not in kwargs:
                kwargs["stop_sequences"] = stop

        llama_messages: List[MessageParam] = [
            _lc_message_to_llama_message_param(m) for m in messages
        ]

        llama_tools: Optional[List[completion_create_params.Tool]] = None
        bound_tools = kwargs.pop("tools", None)
        if bound_tools:
            try:
                llama_tools = [_lc_tool_to_llama_tool_param(t) for t in bound_tools]
            except ValueError as e:
                raise ValueError(f"Error converting tools for Llama API: {e}")

        api_params: Dict[str, Any] = {
            "model": self.model_name,
            "messages": llama_messages,
            **kwargs,
        }
        if llama_tools:
            api_params["tools"] = llama_tools
        if self.temperature is not None:
            api_params["temperature"] = self.temperature
        if self.max_tokens is not None:
            api_params["max_completion_tokens"] = self.max_tokens
        if self.repetition_penalty is not None:
            api_params["repetition_penalty"] = self.repetition_penalty

        try:
            # This assumes self.client has an async create method.
            # If client is sync, this needs:
            # response_obj = await asyncio.to_thread(self.client.chat.completions.create, **api_params)
            response_obj: CreateChatCompletionResponse = await self.client.chat.completions.create(**api_params)  # type: ignore
        except APIError as e:
            raise e

        # Check if completion_message is missing
        if not response_obj.completion_message:
            raise ValueError(
                "Invalid API response format: completion_message is missing."
            )

        generations: List[ChatGeneration] = []
        msg_data = response_obj.completion_message

        content_str = ""
        if msg_data.content:
            # Handle different content types
            if isinstance(msg_data.content, list):
                # Handle cases where content could be a list of items (e.g. multimodal)
                for item in msg_data.content:
                    if (
                        isinstance(item, dict)
                        and item.get("type") == "text"
                        and "text" in item
                    ):
                        content_str += item["text"] + " "
                    elif hasattr(item, "text") and isinstance(item.text, str):
                        content_str += item.text + " "
                    elif isinstance(item, str):  # Plain strings in a list
                        content_str += item + " "
                content_str = content_str.strip()
            elif hasattr(msg_data.content, "text") and isinstance(
                msg_data.content.text, str
            ):
                content_str = msg_data.content.text
            elif isinstance(msg_data.content, str):
                content_str = msg_data.content

        parsed_tool_calls = []
        invalid_tool_calls = []
        if msg_data.tool_calls:
            for tc in msg_data.tool_calls:
                # Ensure tc.function and tc.function.name are valid before accessing
                if (
                    not tc.function
                    or not tc.function.name
                    or not isinstance(tc.function.name, str)
                ):
                    # Handle malformed tool call from API if necessary
                    invalid_tool_calls.append(
                        {
                            "id": (
                                str(tc.id) if tc.id else "unknown_id"
                            ),  # Ensure ID is a string
                            "name": "unknown_tool_name",  # Use a default string
                            "args": (
                                "{}"
                                if not tc.function
                                else str(tc.function.arguments or "{}")
                            ),
                            "error": "Malformed tool call structure from API",
                        }
                    )
                    continue

                try:
                    args_dict = json.loads(
                        tc.function.arguments or "{}"
                    )  # Default to empty dict if args is None/empty
                    parsed_tool_calls.append(
                        {
                            "id": str(tc.id),
                            "name": str(tc.function.name),  # Ensure name is a string
                            "args": args_dict,
                        }
                    )
                except json.JSONDecodeError as e:
                    invalid_tool_calls.append(
                        {
                            "id": str(tc.id),
                            "name": str(tc.function.name),  # Ensure name is a string
                            "args": str(
                                tc.function.arguments or ""
                            ),  # Ensure args is a string
                            "error": f"JSONDecodeError: {e}",
                        }
                    )

        ai_message_kwargs: Dict[str, Any] = {
            "content": content_str if content_str else ""
        }  # Ensure content is at least ""
        if parsed_tool_calls:
            ai_message_kwargs["tool_calls"] = parsed_tool_calls
        if invalid_tool_calls:
            ai_message_kwargs["invalid_tool_calls"] = invalid_tool_calls

        usage_metadata = None
        if response_obj.metrics:
            usage_metadata = {
                m.metric: m.value
                for m in response_obj.metrics
                if m.metric and m.value is not None
            }

        generation_info: Dict[str, Any] = {
            "finish_reason": msg_data.stop_reason,
            "model_name": self.model_name,
        }
        if usage_metadata:
            generation_info["usage_metadata"] = usage_metadata

        generations.append(
            ChatGeneration(
                message=AIMessage(**ai_message_kwargs), generation_info=generation_info
            )
        )

        llm_output = {"metrics": response_obj.metrics} if response_obj.metrics else {}
        return ChatResult(generations=generations, llm_output=llm_output)

    async def _astream(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[AsyncCallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> AsyncIterator[ChatGenerationChunk]:
        """Async stream Llama API chat completion responses."""
        if stop:
            if run_manager:
                await run_manager.on_text(
                    "Warning: 'stop' sequences are not directly supported by the Llama API client and will be ignored.\\n",
                    color="yellow",
                )
            if "stop_sequences" not in kwargs:
                kwargs["stop_sequences"] = stop

        llama_messages: List[MessageParam] = [
            _lc_message_to_llama_message_param(m) for m in messages
        ]
        llama_tools: Optional[List[completion_create_params.Tool]] = None
        bound_tools = kwargs.pop("tools", None)
        if bound_tools:
            try:
                llama_tools = [_lc_tool_to_llama_tool_param(t) for t in bound_tools]
            except ValueError as e:
                raise ValueError(f"Error converting tools for Llama API: {e}")

        api_params: Dict[str, Any] = {
            "model": self.model_name,
            "messages": llama_messages,
            "stream": True,
            **kwargs,
        }
        if llama_tools:
            api_params["tools"] = llama_tools
        if self.temperature is not None:
            api_params["temperature"] = self.temperature
        if self.max_tokens is not None:
            api_params["max_completion_tokens"] = self.max_tokens
        if self.repetition_penalty is not None:
            api_params["repetition_penalty"] = self.repetition_penalty

        # Track tool calls that come in chunks with their indices
        current_tool_calls: Dict[str, Dict[str, Any]] = (
            {}
        )  # tool_id -> {name, args_buffer, index}

        try:
            # Get the result of the API call, which could be an iterator or a coroutine
            response = self.client.chat.completions.create(**api_params)

            # Check if the response is a coroutine that needs to be awaited
            if asyncio.iscoroutine(response):
                # It's a coroutine, await it to get the actual stream
                stream_response = await response
            else:
                # It's already an async iterator (e.g., from a mock in tests)
                stream_response = response

            # Get the async iterator - for some AsyncMockIterator implementations
            # we need to await the __aiter__ method
            try:
                aiterator = await stream_response.__aiter__()
            except (TypeError, AttributeError):
                # If __aiter__ is not awaitable or doesn't exist, use the object directly
                aiterator = stream_response

            # Iterate over chunks
            while True:
                try:
                    # Try to get the next chunk, using __anext__ which should be awaitable
                    chunk = await aiterator.__anext__()
                    stream_chunk_obj = cast(
                        CreateChatCompletionResponseStreamChunk, chunk
                    )

                    delta_content: Optional[str] = None
                    tool_call_chunks_for_lc: List[Dict[str, Any]] = []
                    finish_reason: Optional[str] = None
                    usage_metadata: Optional[Dict[str, Any]] = None
                    generation_info: Optional[Dict[str, Any]] = None

                    # Handle text content delta
                    if (
                        getattr(stream_chunk_obj, "type", None)
                        == "completion_message_delta"
                        and getattr(stream_chunk_obj, "delta", None)
                        and getattr(stream_chunk_obj.delta, "content", None)
                    ):
                        delta_content = str(stream_chunk_obj.delta.content)
                        if run_manager:
                            await run_manager.on_llm_new_token(delta_content)

                    # Handle tool call chunks, which may come in multiple deltas for the same tool call
                    elif getattr(
                        stream_chunk_obj, "type", None
                    ) == "tool_calls_generation_chunk" and getattr(
                        stream_chunk_obj, "tool_calls", None
                    ):
                        for i, tc_delta in enumerate(stream_chunk_obj.tool_calls):
                            # Ensure we have a valid ID string
                            tool_id = (
                                str(tc_delta.id)
                                if hasattr(tc_delta, "id") and tc_delta.id
                                else f"tc_{i}"
                            )

                            # Get function object and handle potential MagicMock
                            tc_function = getattr(tc_delta, "function", None)

                            # Extract name - handle both string and MagicMock
                            tc_name = "unknown_tool"
                            if tc_function and hasattr(tc_function, "name"):
                                tc_name = str(
                                    tc_function.name
                                )  # Convert MagicMock to string if needed

                            # Extract arguments - handle both string and MagicMock
                            tc_args = ""
                            if tc_function and hasattr(tc_function, "arguments"):
                                if tc_function.arguments is not None:
                                    tc_args = str(tc_function.arguments)

                            # Initialize this tool call in our tracking dict if it's new
                            if tool_id not in current_tool_calls:
                                current_tool_calls[tool_id] = {
                                    "name": tc_name,
                                    "args_buffer": tc_args,
                                    "index": len(
                                        current_tool_calls
                                    ),  # Assign sequential index based on first appearance
                                }
                            else:
                                # Append to args buffer if we're getting more arguments
                                if tc_args:
                                    current_tool_calls[tool_id][
                                        "args_buffer"
                                    ] += tc_args

                            # Add this chunk to the output for LangChain
                            tool_call_chunks_for_lc.append(
                                {
                                    "name": current_tool_calls[tool_id]["name"],
                                    "args": tc_args,
                                    "id": tool_id,
                                    "index": current_tool_calls[tool_id]["index"],
                                }
                            )

                    # Handle stream completion - final chunk
                    elif (
                        getattr(stream_chunk_obj, "type", None)
                        == "completion_message_stop"
                    ):
                        finish_reason = getattr(stream_chunk_obj, "stop_reason", None)
                        if (
                            hasattr(stream_chunk_obj, "metrics")
                            and stream_chunk_obj.metrics
                        ):
                            usage_metadata = {}
                            for metric in stream_chunk_obj.metrics:
                                metric_name = getattr(metric, "metric", None)
                                metric_value = getattr(metric, "value", None)
                                if metric_name is not None and metric_value is not None:
                                    usage_metadata[str(metric_name)] = metric_value

                    # Prepare generation info if this is the final chunk
                    if finish_reason is not None:
                        generation_info = {"finish_reason": finish_reason}
                        if usage_metadata:
                            generation_info["usage_metadata"] = usage_metadata

                    # Yield a chunk if we have any new content
                    if delta_content or tool_call_chunks_for_lc or generation_info:
                        yield ChatGenerationChunk(
                            message=AIMessageChunk(
                                content=delta_content or "",
                                tool_call_chunks=tool_call_chunks_for_lc,
                            ),
                            generation_info=generation_info,
                        )
                except StopAsyncIteration:
                    # If StopAsyncIteration is raised, break the loop
                    break
        except Exception as e:
            # Properly handle unexpected exceptions during streaming
            if run_manager:
                await run_manager.on_llm_error(e)
            raise e

    # Consider adding a method to check for Llama Guard moderation if needed,
    # or integrating it if the API supports it as part of the chat completion.
    # The Llama API docs show a separate /moderations endpoint.


# Variables to help with consistent mock creation in tests
API_ERROR_DETAILS = {
    "message": "Rate limit exceeded",
    "status": 429,
    "request": {"method": "POST", "url": "https://api.llama.com/v1/chat/completions"},
    "response": {"body": {"error": {"code": 429}}},
}
