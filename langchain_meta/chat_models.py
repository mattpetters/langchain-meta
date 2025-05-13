# https://python.langchain.com/docs/how_to/custom_chat_model/

import json
import logging
import os
import warnings
from typing import (
    Any,
    Callable,
    ClassVar,
    Dict,
    List,
    Literal,
    Optional,
    Sequence,
    Type,
    Union,
    get_type_hints,
)

from langchain_core.language_models.base import LanguageModelInput
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    SystemMessage,
)
from langchain_core.output_parsers.openai_tools import (
    JsonOutputKeyToolsParser,
    PydanticToolsParser,
)
from langchain_core.runnables import Runnable, RunnableLambda, RunnablePassthrough
from langchain_core.tools import BaseTool
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

# Import the mixin
from langchain_meta.chat_meta_llama.chat_async import AsyncChatMetaLlamaMixin

from .chat_meta_llama.chat_sync import SyncChatMetaLlamaMixin
from .chat_meta_llama.serialization import (
    _lc_message_to_llama_message_param,
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
    model_name: Optional[str] = Field(default=LLAMA_DEFAULT_MODEL_NAME, alias="model")

    # Optional parameters for the Llama API, with LangChain common names where applicable
    temperature: Optional[float] = Field(default=None)  # Added default
    max_tokens: Optional[int] = Field(
        default=None, alias="max_completion_tokens"
    )  # LangChain uses max_tokens
    repetition_penalty: Optional[float] = Field(default=None)  # Added default
    stop: Optional[List[str]] = Field(default=None) # For LangSmith compatibility

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
        *,  # Make all args keyword-only
        model_name: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        repetition_penalty: Optional[float] = None,
        llama_api_key: Optional[str] = None,
        llama_api_url: Optional[str] = None,
        stop: Optional[List[str]] = None, # Added stop param
        client: Optional[LlamaAPIClient] = None,
        async_client: Optional[AsyncLlamaAPIClient] = None,
        **kwargs: Any,
    ):
        # Separate known Llama API params from other kwargs that might be intended for model_kwargs
        init_kwargs = {}
        remaining_kwargs = {}
        known_fields = self.model_fields.keys()

        for key, value in kwargs.items():
            if key in known_fields:
                init_kwargs[key] = value
            else:
                remaining_kwargs[key] = value

        # Always pop max_tokens from remaining_kwargs before passing to super to avoid double-passing if it was in **kwargs
        # The direct max_tokens arg is handled by Pydantic field itself.
        if max_tokens is not None and "max_tokens" in remaining_kwargs:
             logger.warning("'max_tokens' was passed both as a direct argument and in **kwargs. Using direct argument.")
             remaining_kwargs.pop("max_tokens", None)
        elif max_tokens is None and "max_tokens" in remaining_kwargs: # max_tokens is in kwargs but not as direct arg
            # Let Pydantic handle it if it's an alias, or it will go into model_kwargs if not a field
            pass # Don't pop here, let super().__init__ or model_kwargs catch it

        super().__init__(
            model_name=model_name, # Let Pydantic handle None via Field default
            temperature=temperature,
            max_tokens=max_tokens, # Pass direct arg
            repetition_penalty=repetition_penalty,
            llama_api_key=llama_api_key,
            llama_api_url=llama_api_url,
            stop=stop, # Pass stop
            client=client,
            async_client=async_client,
            **init_kwargs # Pass known fields from original kwargs
        )
        # Initialize model_kwargs with any remaining (unconsumed) keyword arguments
        # This ensures that if BaseChatModel also has a model_kwargs, ours takes precedence
        # or that we initialize it if BaseChatModel doesn't.
        # We also add any kwargs that were not model fields of ChatMetaLlama
        current_model_kwargs = getattr(self, "model_kwargs", {}) # Get if super already set it
        if not isinstance(current_model_kwargs, dict): # Ensure it's a dict
            current_model_kwargs = {}
        current_model_kwargs.update(remaining_kwargs) # Add our remaining kwargs
        self.model_kwargs = current_model_kwargs

        self._client = client
        self._async_client = async_client

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
        tools: Optional[List[completion_create_params.Tool]] = None,
        stop: Optional[List[str]] = None,
        stream: bool = False,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """Prepares API parameters for a chat completion request."""
        api_params: Dict[str, Any] = {
            "model": self.model_name,
            "messages": [
                _lc_message_to_llama_message_param(m) for m in messages
            ],  # Convert messages properly
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
                # Check if the parameter is generally supported by the ChatMetaLlama class
                # before adding it to API params. Note: This still relies on SUPPORTED_PARAMS
                # which includes max_completion_tokens, but we are manually excluding it here.
                if api_param_name in self.SUPPORTED_PARAMS:
                    api_params[api_param_name] = kwargs.pop(key)

        # Handle max_tokens (alias max_completion_tokens for Llama API)
        # Prefer max_tokens from direct call (via kwargs) > self.max_tokens
        max_tokens_val = kwargs.pop("max_tokens", self.max_tokens)
        if max_tokens_val is not None:
            api_params["max_completion_tokens"] = max_tokens_val

        # Add tools if provided
        if tools:
            api_params["tools"] = tools
            # If tools are present and no specific tool_choice is given, set to "auto"
            # to encourage the model to use the tools.
            if (
                "tool_choice" not in api_params
                and kwargs.get("tool_choice", None) is None
            ):
                api_params["tool_choice"] = "auto"
                logger.debug(
                    "Set tool_choice='auto' as tools are present and no specific choice was made."
                )

        # Stop sequence is not directly supported by the client's create method, so we don't add it here.
        if stop:
            logger.warning(
                "'stop' sequences were provided, but are not directly supported by the Llama API client's create method and will be ignored."
            )

        # Add response_format if provided (for structured output)
        if "response_format" in kwargs:
            # Meta uses response_format parameter for json_schema output
            api_params["response_format"] = kwargs.pop("response_format")

        # Check for any remaining kwargs that are not supported and warn
        # Add max_tokens explicitly to the list of ignored keys here since it's a known unsupported param for the client
        IGNORED_PARAMS = [
            "client",
            "async_client",
            "run_manager",
            "callbacks",
            "max_tokens",
            "system_prompt",  # Handled via messages, not directly
        ]
        for key in kwargs.keys():
            # Also check if the key is in SUPPORTED_PARAMS but is one we are explicitly excluding for this client version
            if key not in self.SUPPORTED_PARAMS and key not in IGNORED_PARAMS:
                logger.warning(
                    f"Unsupported parameter passed to API call: {key}. It will be ignored."
                )

        # Ensure tool_choice is never sent to the Llama API endpoint
        api_params.pop("tool_choice", None)

        return api_params

    def _get_invocation_params(self, api_params: Optional[Dict[str, Any]] = None, **kwargs: Any) -> Dict[str, Any]:
        """Return a dictionary of parameters used for invocation.

        This is used by the on_chat_model_start callback.
        This should include all model parameters, plus provider-specific parameters.
        Passed to the callback manager's on_chat_model_start method.
        """
        params = {
            # Start with base identifying params
            **self._identifying_params, # Core model params like name, temp
            # Merge model_kwargs (additional init args)
            **(self.model_kwargs or {}), # Other model-specific params from init
            # Merge kwargs passed directly to generate/invoke call
            **(kwargs or {}),
            # Merge api_params (less critical for this callback, might be None)
            **(api_params or {}),
        }
        # Ensure correct mapping for tracing, e.g., max_tokens vs max_completion_tokens
        if "max_completion_tokens" in params and "max_tokens" not in params:
            params["max_tokens"] = params["max_completion_tokens"]

        # --- Add ls_structured_output_format metadata to params['options'][0] --- 
        is_json_mode = (
            isinstance(params.get("response_format"), dict)
            and params["response_format"].get("type") == "json_schema"
        )
        is_function_calling = bool(params.get("tools"))

        if is_json_mode or is_function_calling:
            schema_dict_for_metadata = None
            method_for_metadata = None
            if is_json_mode:
                method_for_metadata = "json_mode"
                schema_holder = params.get("response_format", {}).get("json_schema", {})
                schema_dict_for_metadata = schema_holder.get("schema")
            elif is_function_calling:
                method_for_metadata = "function_calling"
                tools_list = params.get("tools", [])
                if tools_list and isinstance(tools_list[0], dict) and "function" in tools_list[0]:
                    schema_dict_for_metadata = tools_list[0]["function"].get("parameters")

            if method_for_metadata:
                structured_output_item = {
                    "ls_structured_output_format": {
                        "schema": schema_dict_for_metadata,
                        "method": method_for_metadata,
                    }
                }
                # Ensure 'options' key exists and is a list
                if "options" not in params or not isinstance(params["options"], list):
                    params["options"] = []
                
                # Remove any existing ls_structured_output_format from options
                params["options"] = [opt for opt in params["options"] 
                                     if not (isinstance(opt, dict) and "ls_structured_output_format" in opt)]
                
                # Insert the new metadata at the beginning of the options list
                params["options"].insert(0, structured_output_item)
                
                # Remove the direct key if it was added in a previous attempt
                params.pop("ls_structured_output_format", None)

        # Remove the client instances if they accidentally got into params
        params.pop("client", None)
        params.pop("async_client", None)
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
        method: Literal["function_calling", "json_mode", "json_schema"] = "function_calling",
        include_raw: bool = False,
        **kwargs: Any,
    ) -> Runnable[LanguageModelInput, Union[Dict, BaseModel]]:
        """
        Return a new runnable that returns a structured output.
        """
        is_pydantic_v2 = False
        is_pydantic_v1 = False
        is_typeddict = False
        schema_name = None
        schema_dict = None
        if isinstance(schema, type):
            # v2: __version__ >= 2, v1: <2
            if hasattr(schema, "model_json_schema"):
                is_pydantic_v2 = True
                schema_name = schema.__name__
                raw_schema_dict = schema.model_json_schema()
                # Ensure all properties have a type, especially for optional fields
                if 'properties' in raw_schema_dict:
                    for prop_name, prop_details in raw_schema_dict['properties'].items():
                        if 'type' not in prop_details:
                            # Try to infer from Pydantic field info if available, or default to string
                            field_info = schema.model_fields.get(prop_name)
                            if field_info and hasattr(field_info, 'annotation'):
                                if field_info.annotation is str: prop_details['type'] = 'string'
                                elif field_info.annotation is int: prop_details['type'] = 'integer'
                                elif field_info.annotation is float: prop_details['type'] = 'number'
                                elif field_info.annotation is bool: prop_details['type'] = 'boolean'
                                elif field_info.annotation is list: prop_details['type'] = 'array'
                                elif field_info.annotation is dict: prop_details['type'] = 'object'
                                else: prop_details['type'] = 'string' # Default
                            else:
                                prop_details['type'] = 'string' # Default if no annotation
                schema_dict = raw_schema_dict
            elif hasattr(schema, "schema"):
                is_pydantic_v1 = True
                schema_name = schema.__name__
                raw_schema_dict = schema.schema()
                # Ensure all properties have a type for Pydantic v1 as well
                if 'properties' in raw_schema_dict:
                    for prop_name, prop_details in raw_schema_dict['properties'].items():
                        if 'type' not in prop_details:
                            # Pydantic v1 field info is different, might need more complex introspection
                            # For now, defaulting to string if type is missing
                            prop_details['type'] = 'string'
                schema_dict = raw_schema_dict
            # --- Add TypedDict Handling Here ---
            elif (
                hasattr(schema, "__annotations__")
                # Check for TypedDict specific attributes more reliably
                and all(hasattr(schema, attr) for attr in ('__required_keys__', '__optional_keys__'))
            ):
                is_typeddict = True
                schema_name = schema.__name__
                hints = get_type_hints(schema)
                properties = {}
                required = list(getattr(schema, '__required_keys__', set()))

                for k, v_type in hints.items():
                    # Determine JSON schema type from Python type
                    if v_type is str: json_type = "string"
                    elif v_type is int: json_type = "integer"
                    elif v_type is float: json_type = "number"
                    elif v_type is bool: json_type = "boolean"
                    elif v_type is dict: json_type = "object"
                    elif v_type is list: json_type = "array"
                    elif hasattr(v_type, '__origin__') and v_type.__origin__ is Union:
                        args = getattr(v_type, '__args__', ())
                        non_none_type = next((arg for arg in args if arg is not type(None)), str)
                        if non_none_type is str: json_type = "string"
                        elif non_none_type is int: json_type = "integer"
                        elif non_none_type is float: json_type = "number"
                        elif non_none_type is bool: json_type = "boolean"
                        else: json_type = "string"
                    else: json_type = "string"
                    properties[k] = {"type": json_type}

                schema_dict = {
                    "title": schema_name,
                    "type": "object",
                    "properties": properties,
                    "required": required,
                }
            # --- End TypedDict Handling ---
        elif isinstance(schema, dict):
            schema_name = schema.get("name", schema.get("title", "OutputSchema"))
            schema_dict = schema
        else:
            raise ValueError(f"Unsupported schema type: {type(schema)}")

        # Default temperature for structured output if not provided in kwargs
        if "temperature" not in kwargs:
            kwargs["temperature"] = 0.1

        llm_for_binding: BaseChatModel = self
        llm_with_schema_binding: Runnable[LanguageModelInput, BaseMessage]

        if method == "json_mode":
            logger.debug(
                f"Binding 'response_format' for Meta's json_schema mode with schema '{schema_name}'"
            )
            if (
                "system_prompt" in kwargs
            ):
                logger.warning(
                    "'system_prompt' kwarg to 'with_structured_output' with 'json_mode' is not "
                    "directly bound as an API parameter. It should be part of input messages."
                )
                kwargs.pop("system_prompt", None)

            bound_params = {
                "response_format": {
                    "type": "json_schema",
                    "json_schema": {
                        "schema": schema_dict,
                    },
                },
                **kwargs,
            }
            llm_with_schema_binding = llm_for_binding.bind(**bound_params)

        elif method == "function_calling":
            tool_definition = {
                "type": "function",
                "function": {
                    "name": schema_name,
                    "description": schema_dict.get(
                        "description", f"Schema for {schema_name}"
                    ),
                    "parameters": schema_dict,
                },
            }
            logger.debug(
                f"Binding 'tools' for function calling with schema '{schema_name}'"
            )
            if (
                "system_prompt" in kwargs
            ):
                logger.warning(
                    "'system_prompt' kwarg to 'with_structured_output' with 'function_calling' is not "
                    "directly bound as an API parameter. It should be part of input messages."
                )
                kwargs.pop("system_prompt", None)

            llm_with_schema_binding = llm_for_binding.bind(
                tools=[tool_definition], **kwargs
            )
        else:
            raise ValueError(f"Unsupported method for structured output: {method}")

        # Setup appropriate output parser
        output_parser: Runnable[BaseMessage, Union[Dict, BaseModel]]
        if method == "json_mode":
            def _parse_json_output(
                message: BaseMessage,
            ) -> Union[Dict, BaseModel]:
                if not isinstance(message, AIMessage):
                    raise TypeError(
                        f"Expected AIMessage for json_mode parsing, got {type(message)}"
                    )
                json_string = message.content
                if not isinstance(json_string, str) or not json_string.strip():
                    if isinstance(json_string, (dict, list)):
                        try:
                            json_string = json.dumps(json_string)
                        except TypeError as e:
                            raise ValueError(
                                f"AIMessage content is not a JSON string or serializable: {json_string}, error: {e}"
                            )
                    else:
                        raise ValueError(
                            f"AIMessage content is not a JSON string for json_mode: {json_string}"
                        )
                try:
                    if is_pydantic_v2 or is_pydantic_v1:
                        return schema.parse_raw(json_string)
                    else:
                        return json.loads(json_string)
                except (
                    json.JSONDecodeError,
                    Exception,
                ) as e:
                    logger.error(
                        f"Failed to parse JSON output for schema '{schema_name}': {e}. Content: '{json_string}'"
                    )
                    raise ValueError(
                        f"Output could not be parsed as {schema_name}: {e}. Received: '{json_string}'"
                    )
            output_parser = RunnableLambda(_parse_json_output)
        else:
            if is_pydantic_v2 or is_pydantic_v1:
                output_parser = PydanticToolsParser(
                    tools=[schema], first_tool_only=True
                )
            else:
                output_parser = JsonOutputKeyToolsParser(
                    key_name=schema_name if schema_name else "parsed_output", # Provide default
                    first_tool_only=True
                )
        if include_raw:
            base_runnable: Runnable[LanguageModelInput, Any] = llm_with_schema_binding | RunnablePassthrough.assign(parsed=output_parser)
        else:
            base_runnable: Runnable[LanguageModelInput, Union[Dict, BaseModel]] = llm_with_schema_binding | output_parser

        # Revert to returning base runnable directly
        return base_runnable
