# https://python.langchain.com/docs/how_to/custom_chat_model/

import json
import logging
import os
import warnings
import uuid
from os import getenv
from typing import (
    Any,
    Callable,
    ClassVar,
    Dict,
    List,
    Literal,
    Optional,
    Sequence,
    Set,
    Tuple,
    Type,
    Union,
    cast,
    get_type_hints,
)

import requests
from langchain_core.callbacks import (
    AsyncCallbackManagerForLLMRun,
    CallbackManagerForLLMRun,
)
from langchain_core.language_models import (
    BaseChatModel,
    LanguageModelInput,
)
from langchain_core.language_models.base import LangSmithParams
from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    HumanMessage,
    SystemMessage,
)
from langchain_core.output_parsers import (
    JsonOutputParser,
    PydanticOutputParser,
    StrOutputParser,
)
from langchain_core.output_parsers.openai_tools import (
    JsonOutputKeyToolsParser,
    JsonOutputToolsParser,
    PydanticToolsParser,
)
from langchain_core.outputs import ChatGeneration, ChatGenerationChunk, ChatResult
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import (
    Runnable,
    RunnableLambda,
    RunnableMap,
    RunnablePassthrough,
)
from langchain_core.tools import BaseTool
from langchain_core.utils.function_calling import (
    convert_to_openai_function,
    convert_to_openai_tool,
)
from llama_api_client import AsyncLlamaAPIClient, LlamaAPIClient
from llama_api_client.types.chat import (
    completion_create_params,
)
from pydantic import (
    BaseModel,
    Field,
    PrivateAttr,
    SecretStr,
    ValidationError,
    ValidationInfo,
    field_validator,
)
from pydantic.v1 import BaseModel as BaseModelV1  # Import Pydantic V1 BaseModel
from typing_extensions import Literal, Type, get_type_hints

# Import the mixin
from langchain_meta.chat_meta_llama.chat_async import AsyncChatMetaLlamaMixin

from .chat_meta_llama.chat_sync import SyncChatMetaLlamaMixin
from .chat_meta_llama.serialization import (
    _lc_message_to_llama_message_param,
    _lc_tool_to_llama_tool_param,
)

logger = logging.getLogger(__name__)

# Valid models for the Llama API
VALID_MODELS = {
    "Llama-4-Scout-17B-16E-Instruct-FP8",
    "Llama-4-Maverick-17B-128E-Instruct-FP8",
    "Llama-3.3-70B-Instruct",
    "Llama-3.3-8B-Instruct",
}

LLAMA_KNOWN_MODELS = {
    "Llama-3.3-70B-Instruct": {
        "model_name": "Llama-3.3-70B-Instruct",
    },
    "Llama-3.3-8B-Instruct": {
        "model_name": "Llama-3.3-8B-Instruct",
    },
    "Llama-4-Scout-17B-16E-Instruct-FP8": {  # Example, adjust as needed
        "model_name": "Llama-4-Scout-17B-16E-Instruct-FP8",
    },
    "Llama-4-Maverick-17B-128E-Instruct-FP8": {  # Example, adjust as needed
        "model_name": "Llama-4-Maverick-17B-128E-Instruct-FP8",
    },
}

LLAMA_DEFAULT_MODEL_NAME = "Llama-4-Maverick-17B-128E-Instruct-FP8"


# STEEXZDdafsdfgasdfg
class ChatMetaLlama(SyncChatMetaLlamaMixin, AsyncChatMetaLlamaMixin, BaseChatModel):
    """
    LangChain ChatModel wrapper for the native Meta Llama API using llama-api-client.

    Key features:
    - Supports tool calling (model-driven, no tool_choice parameter).
    - Handles message history and tool execution results.
    - Provides streaming and asynchronous generation.
    - Fully compatible with LangSmith tracing and monitoring.

    Differences from OpenAI client:
    - No `tool_choice` parameter to force tool use.
    - Response structure is `response.completion_message` instead of `response.choices[0].message`.
    - `ToolCall` objects in the response do not have a direct `.type` attribute.

    To use, you need to have the `llama-api-client` Python package installed and
    configure your Meta Llama API key and base URL.
    Example:
        ```python
        from llama_api_client import LlamaAPIClient
        from langchain_meta import ChatMetaLlama

        client = LlamaAPIClient(
            api_key=os.environ.get("META_API_KEY"),
            base_url=os.environ.get("META_API_BASE_URL", "https://api.llama.com/v1/")
        )
        llm = ChatMetaLlama(client=client, model_name="Llama-4-Maverick-17B-128E-Instruct-FP8")

        # Basic invocation
        response = llm.invoke([HumanMessage(content="Hello Llama!")])
        print(response.content)

        # Tool calling
        from langchain_core.tools import tool
        @tool
        def get_weather(location: str) -> str:
            '''Gets the current weather in a given location.'''
            return f"The weather in {location} is sunny."

        llm_with_tools = llm.bind_tools([get_weather])
        response = llm_with_tools.invoke("What is the weather in London?")
        print(response.tool_calls)
        ```

    LangSmith integration:
        To enable LangSmith tracing, set these environment variables:
        ```
        LANGSMITH_TRACING=true
        LANGSMITH_API_KEY="your-api-key"
        LANGSMITH_ENDPOINT="https://api.smith.langchain.com"
        LANGSMITH_PROJECT="your-project-name"
        ```
    """

    _client: LlamaAPIClient | None = PrivateAttr(default=None)
    _async_client: AsyncLlamaAPIClient | None = PrivateAttr(default=None)

    # Ensure Pydantic handles the default value if model_name is not provided.
    # The field_validator can then focus on other forms of validation if needed.
    model_name: Optional[str] = Field(default=LLAMA_DEFAULT_MODEL_NAME)

    # Optional parameters for the Llama API, with LangChain common names where applicable
    temperature: Optional[float] = Field(default=None)  # Added default
    max_tokens: Optional[int] = Field(
        default=None, alias="max_completion_tokens"
    )  # LangChain uses max_tokens
    repetition_penalty: Optional[float] = Field(default=None)  # Added default
    stop: Optional[List[str]] = Field(default=None)  # For LangSmith compatibility

    # API Key and Base URL for client initialization if client is not passed
    llama_api_key: Optional[SecretStr] = Field(default=None, alias="api_key")
    llama_api_url: Optional[str] = Field(default=None, alias="base_url")

    # To store any additional keyword arguments passed during initialization
    model_kwargs: Dict[str, Any] = Field(default_factory=dict)

    SUPPORTED_PARAMS: ClassVar[set] = {
        "model",
        "messages",
        "temperature",
        "max_completion_tokens",
        "tools",
        "stream",
        "repetition_penalty",
        "top_p",
        "top_k",
        "user",
        "response_format",
    }

    model_config = {
        "validate_assignment": True,
        "validate_by_name": True,
    }

    def __init__(
        self,
        *,
        model_name: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        repetition_penalty: Optional[float] = None,
        llama_api_key: Optional[str] = None,
        llama_api_url: Optional[str] = None,
        stop: Optional[List[str]] = None,
        client: Optional[LlamaAPIClient] = None,
        async_client: Optional[AsyncLlamaAPIClient] = None,
        **kwargs: Any,
    ):
        # Collect all relevant args for super().__init__
        init_args = {
            "model_name": model_name,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "repetition_penalty": repetition_penalty,
            "llama_api_key": llama_api_key,  # Let Pydantic handle SecretStr
            "llama_api_url": llama_api_url,
            "stop": stop,
            **kwargs,  # Pass remaining kwargs as well
        }

        # Filter out None values that are not meant to override Pydantic defaults
        # Only include args that were actually provided (not None)
        filtered_init_args = {k: v for k, v in init_args.items() if v is not None}

        # Call super().__init__ with the arguments that were actually provided.
        # Pydantic will use Field defaults for arguments not present in filtered_init_args.
        super().__init__(**filtered_init_args)

        # Handle model_kwargs *after* super init
        # Any remaining kwargs not assigned to fields by Pydantic are stored here.
        # Note: Pydantic v2 usually handles extra kwargs based on model_config['extra']
        # We might need to refine how remaining_kwargs are identified if using strict models.
        self.model_kwargs = {
            k: v for k, v in kwargs.items() if k not in self.model_fields
        }

        # Assign explicitly passed clients (will override initialization if provided)
        self._client = client
        self._async_client = async_client

        # Ensure clients are initialized if not provided
        # This needs the final values of llama_api_key and llama_api_url set by super init
        self._ensure_client_initialized()

    @field_validator("model_name", mode="before")
    @classmethod
    def validate_model_name(
        cls, v: Any, info: ValidationInfo
    ):  # Changed FieldValidationInfo to ValidationInfo
        # This validator now primarily ensures that if a model_name is provided, it's valid.
        # The default is handled by Pydantic's Field definition.
        if v is None:  # If None comes in (e.g. explicit model_name=None)
            # Pydantic should have already applied the default from Field() if no value was passed.
            # If v is None here, it means it was EXPLICITLY passed as None.
            # In this case, we let Pydantic's default mechanism (from Field) take over,
            # or if the field were truly Optional without a default, None would be fine.
            # Given our Field has a default, this path implies we want that default.
            # Returning None here will let Pydantic use the Field default.
            return None  # Allow Pydantic to use Field's default

        v_str = str(v).strip()
        if not v_str:  # If it's an empty string after strip
            # If an empty string was explicitly passed, fall back to default.
            default_model = LLAMA_DEFAULT_MODEL_NAME  # Field default won't be used if empty str passed
            logger.warning(f"model_name was empty. Defaulting to {default_model}")
            return default_model

        if v_str not in LLAMA_KNOWN_MODELS:
            warnings.warn(
                f"Model \\'{v_str}\\' is not in the list of known Llama models.\\n"
                f"Known models: {', '.join(LLAMA_KNOWN_MODELS.keys())}\\n"
                "Your model may still work if the Meta API accepts it, but hasn\\'t been tested."
            )
        return v_str

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

    def _get_ls_params(self, **kwargs) -> Dict[str, Any]:
        """Return standard params for LangSmith tracing/tests."""
        # These keys are required by the standard tests
        return {
            "ls_provider": "metallama",
            "ls_model_name": self.model_name,
            "ls_model_type": "chat",
            "ls_temperature": self.temperature,
            "ls_max_tokens": self.max_tokens,
            "ls_stop": self.stop,  # Ensure ls_stop is always present
        }

    def _ensure_client_initialized(self) -> None:
        # Retrieve the plain string API key from the SecretStr field.
        key_val = self.llama_api_key.get_secret_value() if self.llama_api_key else None

        # self.llama_api_url is a string field with a Pydantic default.
        url_val = self.llama_api_url  # Corrected from llama_base_url

        if self._client is None:
            if not key_val:  # Check the actual string value of the key
                # Log instead of raising, to match previous behavior pattern for missing optional clients
                logger.warning(
                    "LlamaAPIClient: API key is missing or empty. "
                    "Sync client cannot be initialized."
                )
            else:
                logger.debug("Instantiating LlamaAPIClient for ChatMetaLlama...")
                self._client = LlamaAPIClient(api_key=key_val, base_url=url_val)
                logger.info("LlamaAPIClient for ChatMetaLlama instantiated.")

        if self._async_client is None:
            if not key_val:  # Check the actual string value of the key
                # Log instead of raising, to match previous behavior pattern for missing optional clients
                logger.warning(
                    "AsyncLlamaAPIClient: API key is missing or empty. "
                    "Async client cannot be initialized."
                )
            else:
                logger.debug("Instantiating AsyncLlamaAPIClient for ChatMetaLlama...")
                self._async_client = AsyncLlamaAPIClient(
                    api_key=key_val, base_url=url_val
                )
                logger.info("AsyncLlamaAPIClient for ChatMetaLlama instantiated.")

    def _detect_supervisor_request(self, messages: List[BaseMessage]) -> bool:
        """Detect if this looks like a supervisor routing request.

        Examines the messages to see if they appear to be a supervisor routing request
        by checking for "route" and "next" keywords in system messages.
        """
        for msg in messages:
            if (
                isinstance(msg, SystemMessage)
                and isinstance(msg.content, str)
                and "route" in msg.content.lower()
                and "next" in msg.content.lower()
            ):
                logger.debug("Supervisor request detected in messages")
                return True
        return False

    def _prepare_api_params(
        self,
        messages: List[BaseMessage],
        # Remove tools from direct args, handle via kwargs from bind_tools
        # tools: Optional[List[completion_create_params.Tool]] = None,
        stop: Optional[List[str]] = None,
        stream: bool = False,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """Prepares API parameters for a chat completion request."""
        logger.debug(f"_prepare_api_params received kwargs: {kwargs}")  # Log all kwargs
        logger.debug(f"_prepare_api_params: tools in kwargs: {kwargs.get('tools')}")
        logger.debug(
            f"_prepare_api_params: tool_choice in kwargs: {kwargs.get('tool_choice')}"
        )

        api_params: Dict[str, Any] = {
            "model": self.model_name,
            "messages": [_lc_message_to_llama_message_param(m) for m in messages],
            "stream": stream,
        }

        # Add parameters from instance, potentially overridden by kwargs
        if self.temperature is not None:
            api_params["temperature"] = self.temperature
        if self.repetition_penalty is not None:
            api_params["repetition_penalty"] = self.repetition_penalty

        # Explicitly add/override parameters from kwargs if provided and supported
        for key in ["temperature", "repetition_penalty", "top_p", "top_k", "user"]:
            if key in kwargs:
                api_param_name = key
                if api_param_name in self.SUPPORTED_PARAMS:
                    api_params[api_param_name] = kwargs.pop(key)

        # Handle max_tokens
        max_tokens_val = kwargs.pop("max_tokens", self.max_tokens)
        if max_tokens_val is not None:
            api_params["max_completion_tokens"] = max_tokens_val

        # Process tools from kwargs (passed by bind_tools)
        tools_list = kwargs.pop("tools", None)
        llama_tools = None
        if tools_list:
            try:
                llama_tools = [
                    _lc_tool_to_llama_tool_param(tool) for tool in tools_list
                ]
                api_params["tools"] = llama_tools
            except Exception as e:
                logger.error(f"Error converting tools for Llama API: {e}")
                # Decide if we should proceed without tools or raise

        # Process tool_choice from kwargs
        tool_choice = kwargs.pop("tool_choice", None)
        if tool_choice:
            # Check if it's already Llama format, else adapt if simple string
            if isinstance(tool_choice, str) and tool_choice in [
                "auto",
                "none",
                "any",
                "required",
            ]:
                api_params["tool_choice"] = tool_choice
            elif isinstance(tool_choice, dict):
                api_params["tool_choice"] = tool_choice  # Assume Llama format dict
            else:
                logger.warning(
                    f"Unsupported tool_choice format: {tool_choice}. Ignoring."
                )
        elif (
            llama_tools and "tool_choice" not in api_params
        ):  # Check if tool_choice wasn't already set
            # If tools are present and no specific tool_choice is given, set to "auto"
            api_params["tool_choice"] = "auto"
            logger.debug(
                "Set tool_choice='auto' as tools are present and no specific choice was made."
            )

        # Stop sequence handling
        if stop:
            logger.warning(
                "'stop' sequences were provided, but are not directly supported by the Llama API client's create method and will be ignored."
            )

        # Add response_format if provided
        if "response_format" in kwargs:
            api_params["response_format"] = kwargs.pop("response_format")

        # Warn about remaining unsupported kwargs
        IGNORED_PARAMS = [
            "client",
            "async_client",
            "run_manager",
            "callbacks",
            "max_tokens",  # Handled above
            "system_prompt",
            "tools",  # Handled above
            "tool_choice",  # Handled above
        ]
        for key in kwargs.keys():
            if key not in self.SUPPORTED_PARAMS and key not in IGNORED_PARAMS:
                logger.warning(
                    f"Unsupported parameter passed to API call: {key}. It will be ignored."
                )

        # REMOVED: Ensure tool_choice is never sent to the Llama API endpoint
        api_params.pop("tool_choice", None)  # Remove unconditionally for now

        # Log the final API params being sent (excluding messages for brevity if needed)
        log_params = {k: v for k, v in api_params.items() if k != "messages"}
        logger.debug(f"Final API params prepared: {log_params}")
        if "messages" in api_params:
            logger.debug(f"Messages count: {len(api_params['messages'])}")
            # Optionally log message roles/content summary if needed

        return api_params

    def _get_invocation_params(
        self, api_params: Optional[Dict[str, Any]] = None, **kwargs: Any
    ) -> Dict[str, Any]:
        """Return a dictionary of parameters used for invocation.

        This is used by the on_chat_model_start callback.
        This should include all model parameters, plus provider-specific parameters.
        Passed to the callback manager's on_chat_model_start method.
        """
        params = {
            # Start with base identifying params
            **self._identifying_params,  # Core model params like name, temp
            # Merge model_kwargs (additional init args)
            **(self.model_kwargs or {}),  # Optional kwargs like retries, timeout
        }

        # Add any kwargs passed in
        if api_params:
            params.update(api_params)

        # Add structured output format schema if present
        schema_info = {}
        method_used = "unknown"
        if "response_format" in kwargs and isinstance(kwargs["response_format"], dict):
            rf = kwargs["response_format"]
            if rf.get("type") == "json_schema" and isinstance(
                rf.get("json_schema"), dict
            ):
                schema_info = rf["json_schema"].get("schema", {})
                method_used = (
                    "json_mode"  # Align with standard test expectations for json_schema
                )
        elif (
            "tools" in kwargs and kwargs["tools"]
        ):  # Fallback check for function calling method
            tools = kwargs["tools"]
            if isinstance(tools, list) and len(tools) > 0:
                if isinstance(tools[0], dict) and "function" in tools[0]:
                    # Ensure we check for function existence before accessing parameters
                    function_dict = tools[0]["function"]
                    if isinstance(function_dict, dict):
                        schema_info = function_dict.get("parameters", {})
                        method_used = "function_calling"

        # Only add if schema_info was populated
        if schema_info:
            params["ls_structured_output_format"] = {
                "schema": schema_info,
                "method": method_used,
            }

        # Merge in any kwargs passed directly to invoke
        params.update({k: v for k, v in kwargs.items() if v is not None})

        return params

    def _count_tokens(self, messages: List[BaseMessage]) -> int:
        """Counts the number of tokens in a list of messages."""
        return sum(len(message.content) for message in messages)

    def _extract_content_from_response(self, response: Any) -> str:
        """Extracts content from a chat completion response."""
        if isinstance(response, dict) and "choices" in response:
            for choice in response["choices"]:
                if "message" in choice and "content" in choice["message"]:
                    return choice["message"]["content"]
        return ""

    def get_token_ids(self, text: str) -> List[int]:
        """Approximate token IDs using character length."""
        # This is a simple fallback. A more accurate method would use a proper tokenizer.
        # For basic testing and fallback, counting characters or simple splitting is sufficient.
        # We return a list of integers to match the expected return type.
        return [
            ord(c) for c in text
        ]  # Using ASCII values as a placeholder for token IDs

    def get_num_tokens(self, text: str) -> int:
        """Get the number of tokens in a text string.

        Uses character count as a simple approximation.
        """
        return len(self.get_token_ids(text))

    def bind_tools(
        self,
        tools: Sequence[Union[Dict[str, Any], Type[BaseModel], Callable, BaseTool]],
        *,
        tool_choice: Optional[Union[str, dict, Literal["any", "auto"]]] = None,
        **kwargs: Any,
    ) -> Runnable[LanguageModelInput, BaseMessage]:  # MODIFIED (removed quotes)
        """
        Bind tool-like objects to this chat model.

        Args:
            tools: A list of tools to bind to the model.
            tool_choice: Optional tool choice.
            **kwargs: Aditional keyword arguments.

        Returns:
            A new Runnable with the tools bound.
        """
        # Correctly delegate to the model's own .bind() method,
        # passing the tools under the 'tools' keyword.
        logger.debug(
            f"ChatMetaLlama.bind_tools called with tools: {[getattr(t, 'name', t) for t in tools]}, tool_choice: {tool_choice}, and kwargs: {kwargs}"
        )
        return self.bind(tools=tools, tool_choice=tool_choice, **kwargs)

    def with_structured_output(
        self,
        schema: Union[Dict, Type[BaseModel]],
        *,
        # Allow json_mode as an alias for json_schema
        method: Literal["function_calling", "json_schema", "json_mode"] = "json_schema",
        include_raw: bool = False,
        **kwargs: Any,
    ) -> Runnable[LanguageModelInput, Union[Dict, BaseModel]]:
        """
        Return a new runnable that returns structured outputs according to the schema.

        This method supports various schema types including Pydantic models (v1 and v2),
        TypedDict, and direct schema dictionaries.

        Args:
            schema: The schema to use for structured output. Can be a Pydantic model
                  class or a dictionary schema.
            method: The method to use for structured output. Defaults to "function_calling".
            include_raw: Whether to include the raw output in the response.
            **kwargs: Additional keyword arguments to pass to the underlying model.

        Returns:
            A runnable that returns structured outputs.
        """
        schema_name = None
        schema_dict = None
        is_pydantic = False
        is_typeddict = False

        # Handle different schema types
        if isinstance(schema, dict):
            # Dictionary schema - get name and dict from it
            schema_dict = schema
            if schema.get("function"):
                schema_name = schema["function"].get("name")
                schema_dict = schema["function"].get("parameters", {})
        elif (
            isinstance(schema, type)
            and (
                issubclass(schema, BaseModel) or issubclass(schema, BaseModelV1)
            )  # Check for V1 or V2
        ):
            # Pydantic model (V1 or V2)
            is_pydantic = True  # Mark that a Pydantic model was processed
            schema_name = schema.__name__
            if issubclass(schema, BaseModelV1):
                # Pydantic V1: Use .schema() to get the schema dictionary
                schema_dict = schema.schema()
            else:
                # Pydantic V2: Use model_to_json_schema helper or schema.model_json_schema()
                schema_dict = model_to_json_schema(schema)
        elif (
            isinstance(schema, type)
            and hasattr(schema, "__annotations__")
            and hasattr(schema, "__required_keys__")
            and hasattr(schema, "__optional_keys__")
        ):
            # TypedDict
            is_typeddict = True
            schema_name = schema.__name__
            # Convert TypedDict to schema
            properties = {}
            required = []

            for name, field_type in get_type_hints(schema).items():
                properties[name] = {"type": "string"}  # Simplifying
                if (
                    hasattr(schema, "__required_keys__")
                    and name in schema.__required_keys__
                ):
                    required.append(name)

            schema_dict = {
                "title": schema_name,
                "type": "object",
                "properties": properties,
                "required": required,
            }
        else:
            raise ValueError(
                f"Schema must be a dictionary, Pydantic model, or TypedDict, got {type(schema)}"
            )

        # Use json_schema mode by default or if explicitly requested
        # Treat json_mode the same as json_schema for Meta Llama API
        if method in ("json_schema", "json_mode"):
            # Construct the response_format parameter for Meta Llama API
            response_format = {
                "type": "json_schema",
                "json_schema": {
                    # Use the extracted name, default if None
                    "name": schema_name or "StructuredOutput",
                    "schema": schema_dict,
                },
            }

            # Bind response_format to the LLM
            bound_llm = self.bind(response_format=response_format, **kwargs)

            # Use appropriate parser for the *content* of the response
            output_parser: Runnable = (
                PydanticOutputParser(pydantic_object=schema)  # type: ignore
                if is_pydantic
                else JsonOutputParser()  # Use standard JSON parser for dict/TypedDict
            )

            chain = bound_llm | output_parser

            if include_raw:
                # RunnableMap takes a dict mapping keys to runnables
                # Pass through the original response and add parsed output
                return RunnableMap({"parsed": chain, "raw": RunnablePassthrough()})
            else:
                return chain

        elif method == "function_calling":
            # Keep the original function calling logic as an alternative method
            # Traditional function calling approach
            # Define a single "output_formatter" function for the LLM to call
            output_function = {
                "name": schema_name or "output_formatter",
                "description": "Format the output according to the schema",
                "parameters": schema_dict,
            }

            # Set up function calling parameters
            params = {
                "tools": [{"type": "function", "function": output_function}],
                "tool_choice": {
                    "type": "function",
                    "function": {"name": output_function["name"]},
                },
            }

            # Create function call processor for the schema
            bound_llm = self.bind(**params)

            # Create parser for tool calls into structured output
            if is_pydantic:
                # Use Pydantic parser for Pydantic models
                # include_raw is handled by RunnableMap wrapper
                parser = PydanticToolsParser(tools=[schema], first_tool_only=True)  # type: ignore
            else:
                # Use JSON key parser for dict schemas or TypedDict
                # args_schema is optional, let parser handle validation if needed
                # include_raw is handled by RunnableMap wrapper
                parser = JsonOutputKeyToolsParser(
                    key_name=output_function["name"],
                    first_tool_only=True,
                )

            chain = bound_llm | parser
            if include_raw:
                # RunnableMap takes a dict mapping keys to runnables
                # Pass through the original response and add parsed output
                return RunnableMap({"parsed": chain, "raw": RunnablePassthrough()})
            else:
                return chain

        # Support other methods here if needed in the future
        else:
            # Should not happen due to Literal type hint, but added for safety
            raise ValueError(f"Unsupported method: {method}")


# Helper function to convert a Pydantic model to a JSON schema
def model_to_json_schema(model_class: Type[BaseModel]) -> Dict[str, Any]:
    """Convert a Pydantic model class to a JSON schema."""
    if hasattr(model_class, "model_json_schema"):
        # Pydantic v2
        return model_class.model_json_schema()
    elif hasattr(model_class, "schema"):
        # Pydantic v1
        return model_class.schema()
    else:
        # Fallback
        properties = {}
        required = []
        for name, field in model_class.__annotations__.items():
            properties[name] = {"type": "string"}  # default type
            required.append(name)

        return {
            "title": model_class.__name__,
            "type": "object",
            "properties": properties,
            "required": required,
        }


# Constants used throughout this module
DEFAULT_MAX_TOKENS = 800
