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

    # API Key and Base URL for client initialization if client is not passed
    llama_api_key: Optional[SecretStr] = Field(default=None, alias="api_key")
    llama_api_url: Optional[str] = Field(default=None, alias="base_url")

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
        "response_format",  # Added for structured output support
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
        client: Optional[LlamaAPIClient] = None,
        async_client: Optional[AsyncLlamaAPIClient] = None,
        **kwargs: Any,
    ):
        # If llama_api_key is not provided, try environment variables
        if llama_api_key is None:
            llama_api_key = os.environ.get("LLAMA_API_KEY") or os.environ.get(
                "META_API_KEY"
            )

        init_values = {
            "model_name": model_name,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "repetition_penalty": repetition_penalty,
            "llama_api_key": llama_api_key,
            "llama_api_url": llama_api_url,
            **kwargs,  # Pydantic will handle client/async_client if passed as kwargs due to field defs
        }

        # Remove None values for fields where Pydantic should use its default
        # or for fields that are truly optional and None is acceptable.
        # For 'model_name', if it's None here, Pydantic will use its Field default.
        init_values_filtered = {k: v for k, v in init_values.items() if v is not None}

        # If model_name was explicitly passed as None, ensure it's not in filtered dict
        # so pydantic uses the field's default. If it was not passed at all (not in kwargs),
        # then it won't be in init_values_filtered either.
        # If it was passed as a string, it will be in init_values_filtered.
        if (
            model_name is None and "model_name" in init_values_filtered
        ):  # Should not happen if logic above is right
            del init_values_filtered["model_name"]

        super().__init__(**init_values_filtered)

        # Assign explicitly passed clients to private attributes
        # These are not Pydantic fields but are managed internally.
        if client is not None:
            self._client = client
        if async_client is not None:
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

        # Add stop if provided
        if stop:
            api_params["stop"] = stop

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

    def _get_invocation_params(self, **kwargs: Any) -> Dict[str, Any]:
        """Gets the parameters for a chat completion invocation."""
        return {
            "model": self.model_name,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "repetition_penalty": self.repetition_penalty,
            "top_p": kwargs.get("top_p", 1.0),
            "top_k": kwargs.get("top_k", 0),
        }

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
        method: Literal["function_calling", "json_mode"] = "function_calling",
        include_raw: bool = False,
        **kwargs: Any,
    ) -> Runnable[LanguageModelInput, Union[Dict, BaseModel]]:
        """
        Return a new runnable that returns a structured output.

        Args:
            schema: The Pydantic model or JSON schema to use for structured output.
            method: The method to use for structured output. Options:
                - 'function_calling': Use tool/function calling format (adapted for Meta).
                                      The prompt should guide the model to use the tool.
                - 'json_mode': Use Meta's native json_schema response_format.
                               The prompt must explicitly ask for JSON output matching the schema.
            include_raw: Whether to include the raw LLM output in the result.
            **kwargs: Additional keyword arguments to pass to the LLM.

        Returns:
            A runnable that outputs structured data.
        """
        schema_name: str
        schema_dict: Dict

        if isinstance(schema, type) and issubclass(schema, BaseModel):
            schema_name = schema.__name__
            schema_dict = schema.model_json_schema()
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
            # For json_mode, the prompt given to the LLM must instruct it to produce JSON.
            # We bind the response_format parameter to tell the Llama API to enforce schema.
            if (
                "system_prompt" in kwargs
            ):  # system_prompt is not a direct API kwarg for bind
                logger.warning(
                    "'system_prompt' kwarg to 'with_structured_output' with 'json_mode' is not "
                    "directly bound as an API parameter. It should be part of input messages."
                )
                kwargs.pop("system_prompt", None)

            bound_params = {
                "response_format": {
                    "type": "json_schema",
                    "json_schema": {
                        # "name": schema_name, # Name is not part of Meta's spec for json_schema content here
                        "schema": schema_dict,
                    },
                },
                **kwargs,  # Other kwargs like temperature
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
            ):  # system_prompt is not a direct API kwarg for bind
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
                    # If content is not a non-empty string, attempt to dump if dict/list, else error
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
                    if isinstance(schema, type) and issubclass(schema, BaseModel):
                        return schema.parse_raw(json_string)
                    else:  # dict schema
                        return json.loads(json_string)
                except (
                    json.JSONDecodeError,
                    ValidationError,
                ) as e:  # ValidationError from Pydantic
                    logger.error(
                        f"Failed to parse JSON output for schema '{schema_name}': {e}. Content: '{json_string}'"
                    )
                    raise ValueError(
                        f"Output could not be parsed as {schema_name}: {e}. Received: '{json_string}'"
                    )

            output_parser = RunnableLambda(_parse_json_output)
        else:  # function_calling
            if isinstance(schema, type) and issubclass(schema, BaseModel):
                output_parser = PydanticToolsParser(
                    tools=[schema], first_tool_only=True
                )
            else:  # dict schema
                output_parser = JsonOutputKeyToolsParser(
                    key_name=schema_name, first_tool_only=True, return_single=True
                )

        if include_raw:
            # Assigns the parsed output to a "parsed" key in the output dict
            return llm_with_schema_binding | RunnablePassthrough.assign(
                parsed=output_parser
            )
        else:
            return llm_with_schema_binding | output_parser
