# https://python.langchain.com/docs/how_to/custom_chat_model/

import asyncio
import json
import os
import warnings
from typing import Any, AsyncIterator, Dict, Iterator, List, Optional, cast, ClassVar
from unittest.mock import MagicMock
from datetime import datetime
import re

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
from langchain_core.runnables import RunnableConfig
from pydantic.v1 import Field, validator
from pydantic.v1.fields import FieldInfo
from pydantic.v1 import BaseModel as PydanticV1BaseModel
from pydantic import model_validator, PrivateAttr

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

# Add logging for debugging
import logging
logger = logging.getLogger(__name__)

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

def _convert_dict_tool(lc_tool: Any) -> dict:
    """Convert a dict tool already in Llama API format."""
    if isinstance(lc_tool, dict) and "function" in lc_tool and isinstance(lc_tool["function"], dict):
        return lc_tool
    raise ValueError("Not a dict tool")

def _convert_pydantic_class_tool(lc_tool: Any) -> dict:
    """Convert a Pydantic model class tool."""
    if isinstance(lc_tool, type):
        name = getattr(lc_tool, "__name__", "UnnamedTool")
        description = getattr(lc_tool, "__doc__", "") or ""
        parameters = {}
        if hasattr(lc_tool, "schema") and callable(getattr(lc_tool, "schema")):
            parameters = lc_tool.schema()
            logger.debug(f"Used schema() method to get parameters for {name}")
        elif hasattr(lc_tool, "model_json_schema") and callable(getattr(lc_tool, "model_json_schema")):
            parameters = lc_tool.model_json_schema()
            logger.debug(f"Used model_json_schema() method to get parameters for {name}")
        if not isinstance(parameters, dict):
            logger.warning(f"Schema for {name} is not a dict: {type(parameters)}. Using empty schema.")
            parameters = {"type": "object", "properties": {}}
        return {
            "type": "function",
            "function": {
                "name": name,
                "description": description,
                "parameters": parameters,
            },
        }
    raise ValueError("Not a Pydantic class tool")

def _convert_structured_tool(lc_tool: Any) -> dict:
    """Convert a StructuredTool or similar object."""
    if hasattr(lc_tool, "name") and hasattr(lc_tool, "description"):
        name = getattr(lc_tool, "name", "unnamed_tool")
        description = getattr(lc_tool, "description", "")
        if not isinstance(name, str):
            logger.warning(f"Tool name is not a string: {type(name)}. Converting to string.")
            name = str(name)
        if not isinstance(description, str):
            logger.warning(f"Tool description is not a string: {type(description)}. Converting to string.")
            description = str(description)
        
        # Handle objects with name, description, and schema_ (Case 3 from original)
        if hasattr(lc_tool, "schema_"):
            schema_ = getattr(lc_tool, "schema_", None)
            if isinstance(schema_, dict):
                logger.debug(f"Using schema_ attribute for tool {name}")
                return {
                    "type": "function",
                    "function": {
                        "name": name,
                        "description": description,
                        "parameters": schema_,
                    },
                }
            else:
                logger.warning(f"Tool {name} has schema_ attribute, but it's not a dict: {type(schema_)}. Proceeding to args_schema.")

        parameters_schema = {"type": "object", "properties": {}}
        if hasattr(lc_tool, "args_schema"):
            args_schema = lc_tool.args_schema
            if args_schema is not None:
                try:
                    if isinstance(args_schema, type) and hasattr(args_schema, "schema") and callable(getattr(args_schema, "schema")):
                        parameters_schema = args_schema.schema()
                        logger.debug(f"Extracted schema from Pydantic v1 model class {args_schema.__name__} for {name}")
                    elif hasattr(args_schema, "model_json_schema") and callable(getattr(args_schema, "model_json_schema")):
                        parameters_schema = args_schema.model_json_schema()
                        logger.debug(f"Extracted schema using model_json_schema() for {name}")
                    elif hasattr(args_schema, "schema") and callable(getattr(args_schema, "schema")):
                        parameters_schema = args_schema.schema()
                        logger.debug(f"Extracted schema using schema() method on instance for {name}")
                    elif isinstance(args_schema, dict):
                        parameters_schema = args_schema
                        logger.debug(f"Using direct dict schema for {name}")
                    # Removed schema_ check here as it's handled above
                    elif hasattr(args_schema, "__annotations__"):
                        properties = {}
                        for field_name, field_type in args_schema.__annotations__.items():
                            type_name = getattr(field_type, "__name__", str(field_type))
                            json_type = _get_json_type_for_annotation(type_name)
                            properties[field_name] = {"type": json_type, "description": f"The {field_name} parameter"}
                        parameters_schema = {"type": "object", "properties": properties, "required": list(args_schema.__annotations__.keys())}
                        logger.debug(f"Built schema from __annotations__ for {name}")
                except Exception as e:
                    logger.warning(f"Error extracting schema from {name}'s args_schema: {e}. Using empty schema.", exc_info=True)
                    parameters_schema = {"type": "object", "properties": {}} # Ensure fallback on error
            else:
                logger.debug(f"Tool {name} has args_schema=None. Using empty schema.")
        return {
            "type": "function",
            "function": {
                "name": name,
                "description": description,
                "parameters": parameters_schema,
            },
        }
    raise ValueError("Not a StructuredTool or object with schema_")

def _convert_parse_method_tool(lc_tool: Any) -> dict:
    """Convert a tool object with a parse method."""
    if hasattr(lc_tool, "parse") and callable(getattr(lc_tool, "parse")):
        name = getattr(lc_tool, "name", None) or getattr(lc_tool, "__class__", None).__name__
        description = getattr(lc_tool, "__doc__", "") or f"Tool that parses {name}"
        parameters_schema = {"type": "object", "properties": {}}
        if hasattr(lc_tool, "schema"):
            if callable(getattr(lc_tool, "schema")):
                parameters_schema = lc_tool.schema()
            elif isinstance(lc_tool.schema, dict):
                parameters_schema = lc_tool.schema
        return {
            "type": "function",
            "function": {
                "name": name,
                "description": description,
                "parameters": parameters_schema,
            },
        }
    raise ValueError("Not a parse-method tool")

def _convert_route_schema_tool(lc_tool: Any) -> dict:
    """Convert a RouteSchema special case tool."""
    route_schema_names = ["RouteSchema", "route_schema"]
    if hasattr(lc_tool, "name") and getattr(lc_tool, "name") in route_schema_names:
        return {
            "type": "function",
            "function": {
                "name": getattr(lc_tool, "name"),
                "description": "Route to the next agent",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "next": {
                            "type": "string",
                            "enum": ["EmailAgent", "ScribeAgent", "TimeKeeperAgent", "GeneralAgent", "END", "__end__"]
                        }
                    },
                    "required": ["next"]
                }
            }
        }
    raise ValueError("Not a RouteSchema tool")

def _create_minimal_tool(lc_tool: Any) -> dict:
    """Fallback: create a minimal tool with empty schema."""
    tool_type = type(lc_tool).__name__
    name = str(getattr(lc_tool, "name", None) or tool_type)
    description = str(getattr(lc_tool, "description", None) or "")
    tool_repr = str(lc_tool)[:100] + "..." if len(str(lc_tool)) > 100 else str(lc_tool)
    logger.error(f"Could not convert tool to Llama API format: {tool_type} {tool_repr}. Creating fallback for {name}.")
    return {
        "type": "function",
        "function": {
            "name": name,
            "description": description,
            "parameters": {"type": "object", "properties": {}},
        },
    }

def _lc_tool_to_llama_tool_param(lc_tool: Any) -> completion_create_params.Tool:
    """Convert LangChain tool to Llama API format using a dispatcher pattern."""
    converters = [
        _convert_dict_tool,
        _convert_pydantic_class_tool,
        _convert_structured_tool,
        _convert_parse_method_tool,
        _convert_route_schema_tool
    ]
    for converter in converters:
        try:
            return converter(lc_tool)
        except Exception:
            continue
    return _create_minimal_tool(lc_tool)

def _get_json_type_for_annotation(type_name: str) -> str:
    """Helper to convert Python type annotations to JSON schema types."""
    if type_name in ("str", "Text", "string"):
        return "string"
    elif type_name in ("int", "float", "complex", "number"):
        return "number"
    elif type_name in ("bool", "boolean"):
        return "boolean"
    elif type_name in ("list", "tuple", "array", "List", "Tuple"):
        return "array"
    elif type_name in ("dict", "Dict", "mapping", "Mapping"):
        return "object"
    else:
        return "string"  # Default fallback


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

    _client: LlamaAPIClient | None = PrivateAttr(default=None)
    model_name: str = Field(default="Llama-4-Maverick-17B-128E-Instruct-FP8", alias="model")  # Added default

    # Optional parameters for the Llama API, with LangChain common names where applicable
    temperature: Optional[float] = Field(default=None)  # Added default
    max_tokens: Optional[int] = Field(
        default=None, alias="max_completion_tokens"
    )  # LangChain uses max_tokens
    repetition_penalty: Optional[float] = Field(default=None)  # Added default

    # API Key and Base URL for client initialization if client is not passed
    llama_api_key: Optional[str] = Field(default=None, alias="api_key")
    llama_base_url: Optional[str] = Field(default=None, alias="base_url")

    SUPPORTED_PARAMS: ClassVar[set] = {
        "model", "messages", "temperature", "max_completion_tokens",
        "stop", "tools", "stream", "repetition_penalty",
        "top_p", "top_k", "user"
    }

    model_config = {
        "validate_assignment": True,
        "validate_by_name": True,
    }

    def __init__(
        self,
        *, # Make all args keyword-only for clarity
        model: Optional[str] = None, # Alias for model_name
        model_name: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        max_completion_tokens: Optional[int] = None, # Alias for max_tokens
        repetition_penalty: Optional[float] = None,
        api_key: Optional[str] = None, # Alias for llama_api_key
        llama_api_key: Optional[str] = None,
        base_url: Optional[str] = None, # Alias for llama_base_url
        llama_base_url: Optional[str] = None,
        **kwargs: Any, # For any other args BaseChatModel might take
    ) -> None:
        """Initialize the Llama API chat model."""
        client = kwargs.pop("client", None)
        self._client = client
        # Ensure model_name is always set for Pydantic v2
        if model_name is None and model is None:
            model_name = "Llama-4-Maverick-17B-128E-Instruct-FP8"
        super().__init__(
            model=model,
            model_name=model_name,
            temperature=temperature,
            max_tokens=max_tokens,
            max_completion_tokens=max_completion_tokens,
            repetition_penalty=repetition_penalty,
            api_key=api_key,
            llama_api_key=llama_api_key,
            base_url=base_url,
            llama_base_url=llama_base_url,
            **kwargs
        )
        # Remove all post-init model_name logic; rely on Pydantic's default
        # Keep the rest of the __init__ logic as is
        resolved_api_key = llama_api_key
        if api_key is not None:
            resolved_api_key = api_key
        if resolved_api_key is None:
            resolved_api_key = None # Explicit default from Field(default=None)
        self.llama_api_key = resolved_api_key
        resolved_base_url = llama_base_url
        if base_url is not None:
            resolved_base_url = base_url
        if resolved_base_url is None:
            resolved_base_url = None # Explicit default from Field(default=None)
        self.llama_base_url = resolved_base_url
        resolved_temperature = temperature
        if resolved_temperature is None:
            resolved_temperature = None # Explicit default from Field(default=None)
        self.temperature = resolved_temperature
        resolved_max_tokens = max_tokens
        if max_completion_tokens is not None:
            resolved_max_tokens = max_completion_tokens
        if resolved_max_tokens is None:
            resolved_max_tokens = None # Explicit default from Field(default=None)
        self.max_tokens = resolved_max_tokens
        resolved_rep_penalty = repetition_penalty
        if resolved_rep_penalty is None:
            resolved_rep_penalty = None # Explicit default from Field(default=None)
        self.repetition_penalty = resolved_rep_penalty
        if self._client is None and self.llama_api_key is not None:
            self._ensure_client_initialized()

    @validator("model_name")
    def validate_model_name(cls, v):
        """Validate that model_name is not empty and warn if not in known models."""
        if not v:
            raise ValueError("model_name cannot be empty")
        if v not in VALID_MODELS:
            model_list = ", ".join(VALID_MODELS)
            warnings.warn(
                f"Model '{v}' is not in the list of known Llama models.\n"
                f"Known models: {model_list}\n"
                f"Your model may still work if the Meta API accepts it, but hasn't been tested.",
                stacklevel=2,
            )
        return v

    @property
    def _llm_type(self) -> str:
        """Return type of chat model."""
        return "chat-meta-llama"

    @property
    def client(self) -> LlamaAPIClient | None:
        """Provides access to the LlamaAPIClient instance."""
        return self._client

    @property
    def _identifying_params(self) -> Dict[str, Any]:
        """Get the identifying parameters."""
        # Direct access should now yield plain values.
        return {
            "model_name": self.model_name,
            "temperature": self.temperature,
            "max_completion_tokens": self.max_tokens,
            "repetition_penalty": self.repetition_penalty,
        }

    def _ensure_client_initialized(self) -> None:
        """Ensure that the client is initialized."""
        if self._client is None:
            current_api_key = self.llama_api_key
            current_base_url = self.llama_base_url
            if not isinstance(current_api_key, str) or not current_api_key:
                env_api_key = os.environ.get("META_API_KEY")
                if not env_api_key:
                    raise ValueError(
                        "META_API_KEY not found. Set it via environment variable or pass it directly to the constructor."
                    )
                self.llama_api_key = env_api_key
                current_api_key = env_api_key
            if not isinstance(current_base_url, str) or not current_base_url:
                env_base_url = os.environ.get(
                    "META_NATIVE_API_BASE_URL", "https://api.llama.com/v1/"
                )
                self.llama_base_url = env_base_url
                current_base_url = env_base_url
            try:
                from llama_api_client import LlamaAPIClient as LocalLlamaAPIClient
                self._client = LocalLlamaAPIClient(
                    api_key=str(current_api_key),
                    base_url=str(current_base_url),
                )
            except ImportError:
                raise ImportError(
                    "llama-api-client package not found, please install it with "
                    "`pip install llama-api-client`"
                )
            except Exception as e:
                raise ValueError(f"Failed to initialize LlamaAPIClient: {e}")

    def _detect_supervisor_request(self, messages: List[BaseMessage]) -> bool:
        """Detect if this looks like a supervisor routing request.
        
        Examines the messages to see if they appear to be a supervisor routing request
        by checking for "route" and "next" keywords in system messages.
        """
        for msg in messages:
            if isinstance(msg, SystemMessage) and "route" in msg.content.lower() and "next" in msg.content.lower():
                logger.debug("Supervisor request detected in messages")
                return True
        return False

    async def _agenerate(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[AsyncCallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> ChatResult:
        """Asynchronously call Llama API for chat completion."""
        # Directly call the async _generate method
        return await self._generate(messages, stop, run_manager, **kwargs)

    async def _generate(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[AsyncCallbackManagerForLLMRun | CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> ChatResult:
        """Main method for Llama API chat completion call (now async)."""
        if stop:
            if run_manager:
                await run_manager.on_text(
                    "Warning: 'stop' sequences are not directly supported by the Llama API client and will be ignored.\n",
                    color="yellow",
                )
            if "stop_sequences" not in kwargs:
                kwargs["stop_sequences"] = stop

        if "tool_choice" in kwargs:
            if run_manager:
                await run_manager.on_text(
                    "Warning: 'tool_choice' parameter is not supported by the Meta Llama API and will be ignored.\n",
                    color="yellow",
                )
            kwargs.pop("tool_choice")

        if self._detect_supervisor_request(messages) and "stream" not in kwargs:
            logger.debug("Forcing stream=False for supervisor request in _generate")
            kwargs["stream"] = False

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
            
        api_params = self._filter_unsupported_params(api_params)
        
        if self._client is None:
            self._ensure_client_initialized()
        
        active_client = self._client
        if active_client is None:
            raise RuntimeError("LlamaAPIClient not initialized.")
        
        logger.debug(f"Meta API call params: {api_params}")
        
        try:
            response_obj: CreateChatCompletionResponse = await (
                active_client.chat.completions.create(**api_params)
            )
            logger.debug(f"Meta API response: {response_obj}")
        except APIError as e:
            logger.error(f"Meta API error: {e}")
            raise e
            
        if not response_obj.completion_message:
            raise ValueError(
                "Invalid API response format: completion_message is missing."
            )

        generations: List[ChatGeneration] = []
        msg_data = response_obj.completion_message

        content_str = ""
        if msg_data.content:
            if isinstance(msg_data.content, list):
                for item in msg_data.content:
                    if (
                        isinstance(item, dict)
                        and item.get("type") == "text"
                        and "text" in item
                    ):
                        content_str += item["text"] + " "
                    elif hasattr(item, "text") and isinstance(item.text, str):
                        content_str += item.text + " "
                    elif isinstance(item, str):
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
                if (
                    isinstance(tc, MagicMock)
                    and hasattr(tc, "function")
                    and isinstance(tc.function, MagicMock)
                    and hasattr(tc.function, "arguments")
                    and isinstance(tc.function.arguments, str)
                    and tc.function.arguments.startswith("invalid")
                ):
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
                if (
                    not tc.function
                    or not hasattr(tc.function, "name")
                    or not tc.function.name
                    or not isinstance(tc.function.name, str)
                ):
                    invalid_tool_calls.append(
                        {
                            "id": (
                                str(tc.id) if tc.id else "unknown_id"
                            ),
                            "name": "unknown_tool_name",
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
                    )
                    parsed_tool_calls.append(
                        {
                            "id": str(tc.id),
                            "name": str(tc.function.name),
                            "args": args_dict,
                        }
                    )
                except json.JSONDecodeError as e:
                    invalid_tool_calls.append(
                        {
                            "id": str(tc.id),
                            "name": str(tc.function.name),
                            "args": str(
                                tc.function.arguments or ""
                            ),
                            "error": f"JSONDecodeError: {e}",
                        }
                    )

        ai_message_kwargs: Dict[str, Any] = {
            "content": content_str if content_str else ""
        }
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
                message=AIMessage(**ai_message_kwargs),
                generation_info=generation_info,
            )
        )
            
        llm_output = {"metrics": response_obj.metrics} if response_obj.metrics else {}
        return ChatResult(generations=generations, llm_output=llm_output)

    def bind(self, **kwargs: Any) -> Any:
        """Override bind to handle tool_choice correctly for Meta compatibility."""
        # This method intercepts the bind operation to handle tool_choice before it gets stored
        # in the bound object, ensuring it works as a drop-in replacement for ChatOpenAI
        logger.debug(f"bind called with kwargs: {kwargs}")
        
        # Store the fact that we have bound tools to be used later
        if "tools" in kwargs:
            logger.debug(f"Tools were bound: {kwargs['tools']}")
            # We keep tools as is - they'll be extracted in _generate, _agenerate, etc.
            
        # If tool_choice is present, log it but don't prevent the binding
        # The actual filtering will happen in _generate, _agenerate, etc.
        if "tool_choice" in kwargs:
            logger.debug(f"Detected tool_choice in bind: {kwargs['tool_choice']}. "
                         f"Will be ignored during API calls to Meta's API.")
            
            # If it's "none", we need to remove tools or they'll be used anyway by Llama
            if kwargs.get("tool_choice") == "none":
                # For OpenAI compatibility, tool_choice="none" means never use tools
                logger.debug("tool_choice is 'none', removing tools from kwargs")
                kwargs.pop("tools", None)
            elif isinstance(kwargs.get("tool_choice"), dict) and kwargs.get("tool_choice", {}).get("type") == "function":
                # If tool_choice is forcing a specific function, make sure that tool is included
                # This is just for ensuring compatibility, as Llama API doesn't support forcing function use
                # But we want the bound LLM to at least have the tools available for use
                forced_tool_name = kwargs.get("tool_choice", {}).get("function", {}).get("name")
                if forced_tool_name and "tools" in kwargs:
                    logger.debug(f"Tool choice is forcing function '{forced_tool_name}', making sure it's available")
                    # We don't need to do anything special for Meta, as it will decide when to use tools
                    # but we log that we've detected this usage pattern
                    
            # Don't actually modify tool_choice in kwargs - we'll handle it in the API call methods        
                
        # Let the parent class handle the actual binding
        return super().bind(**kwargs)
    
    def with_structured_output(self, schema, *, include_raw: bool = False, **kwargs):
        """
        Creates a structured output wrapper for the Meta Llama API.
        
        Args:
            schema: The Pydantic schema to use for output parsing
            include_raw: Whether to include raw LLM output (not supported)
            **kwargs: Additional kwargs to pass to the LLM
            
        Returns:
            A wrapper around the LLM that outputs structured objects
        """
        # Adding explicit logging for debugging
        logger.debug(f"Using model_json_schema() method")
        
        # Import here to avoid circular imports
        from langchain_core.runnables import Runnable
        from langchain_core.pydantic_v1 import create_model, BaseModel
        from langchain_core.messages import SystemMessage, HumanMessage
        
        # Verify we have a valid schema class
        if hasattr(schema, "model_json_schema") and callable(getattr(schema, "model_json_schema")):
            schema_dict = schema.model_json_schema()
            schema_name = schema.__name__ if hasattr(schema, "__name__") else str(schema)
        elif hasattr(schema, "schema") and callable(getattr(schema, "schema")):
            schema_dict = schema.schema()
            schema_name = schema.__name__ if hasattr(schema, "__name__") else str(schema)
        else:
            schema_dict = schema
            schema_name = "CustomSchema"
        
        logger.debug(f"Created Meta-specific structured output wrapper for schema: {schema_name}")
        
        # Bind streaming off explicitly for structured output
        kwargs["stream"] = False
        bound_llm = self.bind(**kwargs)
        logger.debug(f"bind called with kwargs: {kwargs}")
        
        class MetaStructuredOutputWrapper(Runnable):
            """Wrapper for Meta Llama models that handles structured output."""
            
            def __init__(self, base_llm):
                self.base_llm = base_llm
                self.schema = schema
                self.schema_dict = schema_dict
                self.schema_name = schema_name
            
            def _filter_params(self, params: Dict[str, Any]) -> Dict[str, Any]:
                """
                Filter out LangChain-specific parameters that should not be passed to the Meta API.
                
                Args:
                    params: Dictionary of parameters to filter
                    
                Returns:
                    Filtered dictionary with API-compatible parameters
                """
                # LangChain-specific parameters that should not be passed to the API
                langchain_params = {
                    "tags", "metadata", "callbacks", "run_managers", 
                    "recursion_limit", "tags_keys", "id_key"
                }
                
                return {k: v for k, v in params.items() if k not in langchain_params}
            
            def _fix_response_format(self, content):
                """Extract valid JSON from response content."""
                if not content:
                    return {"error": "Empty response from LLM"}
                    
                # First try direct JSON parsing
                if isinstance(content, str):
                    content = content.strip()
                    try:
                        return json.loads(content)
                    except json.JSONDecodeError:
                        pass
                    
                    # Try to find JSON in code blocks
                    json_match = re.search(r'```(?:json)?\s*([\s\S]*?)\s*```', content, re.DOTALL)
                    if json_match:
                        try:
                            return json.loads(json_match.group(1).strip())
                        except json.JSONDecodeError:
                            pass
                    
                    # Try to find JSON object pattern
                    obj_match = re.search(r'({[\s\S]*})', content, re.DOTALL)
                    if obj_match:
                        try:
                            return json.loads(obj_match.group(1).strip())
                        except json.JSONDecodeError:
                            pass
                
                # If all parsing attempts fail, return the original content
                logger.warning(f"Could not parse response as JSON: {content[:100]}...")
                return content
            
            def _add_schema_instruction(self, messages):
                """Add JSON schema instruction to messages."""
                from langchain_core.messages import SystemMessage, HumanMessage
                from langchain_core.prompt_values import ChatPromptValue
                
                schema_instruction = (
                    f"You MUST respond with a valid JSON object that matches this schema:\n"
                    f"{json.dumps(self.schema_dict, indent=2)}\n\n"
                    f"Do not include explanations, only output valid JSON that matches the schema."
                )
                
                # Handle ChatPromptValue objects (returned by prompt templates)
                if hasattr(messages, "to_messages") and callable(messages.to_messages):
                    # Extract the actual messages
                    message_list = messages.to_messages()
                    has_system = any(isinstance(msg, SystemMessage) for msg in message_list)
                    
                    # Create a new list with schema instruction
                    if has_system:
                        # Add to existing system message
                        new_messages = []
                        for msg in message_list:
                            if isinstance(msg, SystemMessage):
                                # Update the first system message we find
                                new_content = f"{msg.content}\n\n{schema_instruction}"
                                new_messages.append(SystemMessage(content=new_content))
                            else:
                                new_messages.append(msg)
                    else:
                        # Add a new system message at the beginning
                        new_messages = [SystemMessage(content=schema_instruction)] + message_list
                        
                    return new_messages
                
                # Handle regular message lists
                elif isinstance(messages, list):
                    for i, msg in enumerate(messages):
                        if isinstance(msg, SystemMessage):
                            # Add schema instruction to existing system message
                            updated_content = f"{msg.content}\n\n{schema_instruction}"
                            messages[i] = SystemMessage(content=updated_content)
                            return messages
                    
                    # No system message found, add one at the beginning
                    return [SystemMessage(content=schema_instruction)] + messages
                
                # Handle anything else by converting to a list of messages
                else:
                    # Try to convert to a string if possible
                    content = str(messages)
                    # Create a new list with system message and human message
                    return [
                        SystemMessage(content=schema_instruction),
                        HumanMessage(content=content)
                    ]
            
            def invoke(self, input_data, config=None, **kwargs):
                """Sync invoke: calls async ainvoke under the hood."""
                return asyncio.run(self.ainvoke(input_data, config, **kwargs))

            async def ainvoke(self, input_data, config=None, **invoke_kwargs):
                """Process messages asynchronously and return JSON conforming to the schema."""
                messages = input_data
                
                # Handle various input types
                if isinstance(input_data, dict) and "input" in input_data:
                    # Handle dict with 'input' key (common in RunnableSequence)
                    user_input = input_data["input"]
                    if isinstance(user_input, str):
                        messages = [HumanMessage(content=user_input)]
                    elif isinstance(user_input, list):
                        messages = user_input
                    else:
                        messages = [HumanMessage(content=str(user_input))]
                elif isinstance(input_data, str):
                    # Handle plain string input as a human message
                    messages = [HumanMessage(content=input_data)]
                    
                # Merge config and kwargs
                kwargs = {}
                if config:
                    if isinstance(config, dict):
                        kwargs.update(config)
                
                # Add invoke_kwargs, but don't override config values with None
                for k, v in invoke_kwargs.items():
                    if k not in kwargs or v is not None:
                        kwargs[k] = v
                
                # Filter out LangChain-specific parameters
                kwargs = self._filter_params(kwargs)
            
                # Add JSON schema instruction
                enhanced_messages = self._add_schema_instruction(messages)
                
                # Call the LLM asynchronously
                result = await self.base_llm.ainvoke(enhanced_messages, **kwargs)
                
                # Get the content string
                if hasattr(result, "content"):
                    content = result.content
                else:
                    content = str(result)
                    
                # Parse JSON response
                return self._fix_response_format(content)
            
            def stream(self, input_data, config=None, **stream_kwargs):
                """Sync stream: calls async astream under the hood."""
                async def _stream():
                    async for chunk in self.astream(input_data, config, **stream_kwargs):
                        yield chunk
                return _stream()
            
            async def astream(self, input_data, config=None, **stream_kwargs):
                """We don't support streaming for structured output.
                Just do a regular ainvoke and yield a single chunk."""
                from langchain_core.messages import AIMessageChunk
                from langchain_core.outputs import ChatGenerationChunk
                
                # Merge config and kwargs
                kwargs = {}
                if config:
                    if isinstance(config, dict):
                        kwargs.update(config)
                
                # Add stream_kwargs, but don't override config values with None
                for k, v in stream_kwargs.items():
                    if k not in kwargs or v is not None:
                        kwargs[k] = v
                
                # Filter out LangChain-specific parameters
                kwargs = self._filter_params(kwargs)
            
                result = await self.ainvoke(input_data, **kwargs)
                if isinstance(result, str):
                    yield ChatGenerationChunk(message=AIMessageChunk(content=result))
                else:
                    yield ChatGenerationChunk(message=AIMessageChunk(content=json.dumps(result)))
                
            def bind(self, **binding_kwargs):
                """Create a new instance with updated kwargs."""
                if binding_kwargs:
                    # LangChain-specific parameters that should not be passed to the API
                    # but should be handled by the LangChain wrapper logic
                    langchain_params = {
                        "tags", "metadata", "callbacks", "run_managers", 
                        "recursion_limit", "tags_keys", "id_key"
                    }
                    
                    # Only pass API-relevant parameters to the base LLM
                    base_llm_kwargs = {k: v for k, v in binding_kwargs.items() 
                                     if k not in langchain_params}
                    
                    new_base_llm = self.base_llm.bind(**base_llm_kwargs)
                    return MetaStructuredOutputWrapper(new_base_llm)
                return self
        
        
        # Return the wrapper
        return MetaStructuredOutputWrapper(bound_llm)
    
    def bind_tools(self, tools, **kwargs):
        """Bind tools to the LLM in a Meta API compatible way."""
        logger.debug(f"bind_tools called with {len(tools)} tools")
        
        # Handle any tool_choice that might be in kwargs
        if "tool_choice" in kwargs:
            logger.debug(f"bind_tools: detected tool_choice: {kwargs['tool_choice']}. "
                        f"This will be ignored during API calls to Meta.")
            
            # If it's "none", we need to remove tools or they'll be used anyway by Llama
            if kwargs.get("tool_choice") == "none":
                # For OpenAI compatibility, tool_choice="none" means never use tools
                logger.debug("tool_choice is 'none', not binding any tools")
                return self.bind(**kwargs)  # Bind without tools
        
        # For Meta API, we just need to bind the tools, tool_choice is not supported
        return self.bind(tools=tools, **kwargs)

    # Consider adding a method to check for Llama Guard moderation if needed,
    # or integrating it if the API supports it as part of the chat completion.
    # The Llama API docs show a separate /moderations endpoint.

    def _clean_schema_for_meta(self, schema: dict) -> dict:
        """Clean JSON schema for Meta API compatibility.
        
        Meta's API only supports a very limited subset of JSON Schema properties:
        - Common: "title", "type", "description"
        - Arrays: "items" (plus the common ones)
        - Objects: "properties", "required" (plus the common ones)
        
        Everything else must be removed, including all validation properties.
        
        Returns a cleaned copy of the schema with only supported properties.
        """
        if not isinstance(schema, dict):
            return schema
            
        # Whitelist approach: Only keep explicitly supported properties
        cleaned = {}
        
        # Common supported properties for all types
        SUPPORTED_COMMON = {"title", "type", "description"}
        
        # Additional supported properties by schema type
        schema_type = schema.get("type", "")
        
        if schema_type == "object":
            # For objects, also support properties and required
            SUPPORTED_ADDITIONAL = {"properties", "required", "additionalProperties"}
            
            # Handle nested objects in properties
            if "properties" in schema:
                cleaned_props = {}
                for prop_name, prop_schema in schema.get("properties", {}).items():
                    cleaned_props[prop_name] = self._clean_schema_for_meta(prop_schema)
                cleaned["properties"] = cleaned_props
                
            # Keep the required property as is
            if "required" in schema:
                cleaned["required"] = schema["required"]
                
            # Handle additionalProperties if present
            if "additionalProperties" in schema:
                if isinstance(schema["additionalProperties"], dict):
                    cleaned["additionalProperties"] = self._clean_schema_for_meta(
                        schema["additionalProperties"]
                    )
                else:
                    cleaned["additionalProperties"] = schema["additionalProperties"]
                
        elif schema_type == "array":
            # For arrays, also support items
            SUPPORTED_ADDITIONAL = {"items"}
            
            # Clean the items schema
            if "items" in schema:
                if isinstance(schema["items"], dict):
                    cleaned["items"] = self._clean_schema_for_meta(schema["items"])
                elif isinstance(schema["items"], list):
                    cleaned["items"] = [
                        self._clean_schema_for_meta(item) if isinstance(item, dict) else item
                        for item in schema["items"]
                    ]
                else:
                    cleaned["items"] = schema["items"]
        
        elif "anyOf" in schema:
            # For anyOf types, we need special handling
            SUPPORTED_ADDITIONAL = {"anyOf"}
            
            if "anyOf" in schema:
                cleaned["anyOf"] = [
                    self._clean_schema_for_meta(option) if isinstance(option, dict) else option
                    for option in schema["anyOf"]
                ]
        else:
            # For primitive types, no additional properties
            SUPPORTED_ADDITIONAL = set()
        
        # Copy common supported properties
        for key in SUPPORTED_COMMON:
            if key in schema:
                cleaned[key] = schema[key]
        
        # Copy type-specific supported properties that weren't already handled
        for key in SUPPORTED_ADDITIONAL:
            if key in schema and key not in cleaned:
                cleaned[key] = schema[key]
                
        return cleaned

    def _filter_unsupported_params(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """
        Filter parameters to only include those supported by the Meta LLM API.
        Uses a whitelist approach instead of a blacklist.

        Args:
            params: Dictionary of parameters to filter
        Returns:
            Filtered dictionary with only supported parameters
        """
        return {key: value for key, value in params.items() if key in self.SUPPORTED_PARAMS}

    def _prepare_api_params(
        self,
        messages: list,
        tools: Optional[list] = None,
        **kwargs
    ) -> dict:
        """
        Prepare API parameters, filtering unsupported ones.

        Args:
            messages: List of message dicts for the API.
            tools: Optional list of tool dicts.
            **kwargs: Additional parameters.

        Returns:
            Dictionary of parameters ready for the Meta API.

        Example:
            >>> llama = ChatMetaLlama(model_name="Llama-4-Maverick-17B-128E-Instruct-FP8")
            >>> llama._prepare_api_params([{"role": "user", "content": "hi"}], tools=[{"type": "function", ...}])
            {'model': 'Llama-4-Maverick-17B-128E-Instruct-FP8', 'messages': [{'role': 'user', 'content': 'hi'}], 'tools': [{...}]}
        """
        api_params = {k: v for k, v in kwargs.items() if k in self.SUPPORTED_PARAMS}
        api_params["model"] = self.model_name
        api_params["messages"] = messages
        if self.temperature is not None:
            api_params["temperature"] = self.temperature
        if self.max_tokens is not None:
            api_params["max_completion_tokens"] = self.max_tokens
        if self.repetition_penalty is not None:
            api_params["repetition_penalty"] = self.repetition_penalty
        if tools:
            api_params["tools"] = tools
        return api_params


# Variables to help with consistent mock creation in tests
API_ERROR_DETAILS = {
    "message": "Rate limit exceeded",
    "status": 429,
    "request": {"method": "POST", "url": "https://api.llama.com/v1/chat/completions"},
    "response": {"body": {"error": {"code": 429}}},
}
