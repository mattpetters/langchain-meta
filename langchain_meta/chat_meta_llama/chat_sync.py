import json
import logging
import re  # Added re
import uuid
from datetime import datetime
from typing import (
    Any,
    Callable,
    Dict,
    Iterator,
    List,
    Literal,
    Optional,
    Type,
    Union,
)

from langchain_core.callbacks.manager import CallbackManagerForLLMRun
from langchain_core.messages import (
    AIMessage,
    AIMessageChunk,
    BaseMessage,  # Used by _lc_message_to_llama_message_param if that was here
    ToolCallChunk,  # For streaming tool calls
    ToolCall,
    SystemMessage,
)
from langchain_core.messages.ai import UsageMetadata
from langchain_core.outputs import (
    ChatGeneration,
    ChatGenerationChunk,
    ChatResult,
    LLMResult,  # For callback in streaming
)
from langchain_core.tools import BaseTool
from llama_api_client import LlamaAPIClient
from llama_api_client.types import CreateChatCompletionResponse
from llama_api_client.types.chat import completion_create_params
from llama_api_client.types.completion_message import (
    CompletionMessage as LlamaCompletionMessage,
)

# from llama_api_client.types.create_chat_completion_response import CreateChatCompletionResponse # Only for async
from pydantic import BaseModel

# Assuming chat_models.py is in langchain_meta.chat_models
# and contains helper functions like _lc_tool_to_llama_tool_param and _prepare_api_params
from .serialization import (
    _lc_tool_to_llama_tool_param,
    _parse_textual_tool_args,
)  # Changed from ..chat_models
from ..utils import parse_malformed_args_string  # Import from main utils
from langchain_core.utils.function_calling import convert_to_openai_tool


def _response_message_to_aimessage(msg):
    # Accepts either a dict or a LlamaAPI CompletionMessage model
    # Returns a LangChain AIMessage
    if hasattr(msg, "content"):
        content = msg.content
        if hasattr(content, "text"):
            content = content.text
        elif isinstance(content, dict) and "text" in content:
            content = content["text"]
    elif isinstance(msg, dict) and "content" in msg:
        content = msg["content"]
        if isinstance(content, dict) and "text" in content:
            content = content["text"]
    else:
        content = ""

    tool_calls = []
    if hasattr(msg, "tool_calls") and getattr(msg, "tool_calls", None):
        tool_calls = getattr(msg, "tool_calls")
    elif isinstance(msg, dict) and "tool_calls" in msg and msg["tool_calls"]:
        tool_calls = msg["tool_calls"]
    # Convert tool_calls to list of dicts if present
    if tool_calls:

        def convert_tool_call(tc):
            if isinstance(tc, dict):
                name = tc.get("function", {}).get("name") or tc.get("name")
                if not name or not isinstance(name, str) or name == "unknown_tool":
                    return None
                arguments = tc.get("function", {}).get("arguments") or tc.get("args")
                args_dict = {}
                if isinstance(arguments, dict):
                    args_dict = arguments
                elif isinstance(arguments, str) and arguments.strip():
                    try:
                        args_dict = json.loads(arguments)
                        if not isinstance(args_dict, dict):
                            args_dict = {"value": args_dict}
                    except Exception:
                        args_dict = {"value": arguments}
                return {
                    "id": tc.get("id"),
                    "name": name,
                    "args": args_dict,
                    "type": "function",
                }
            else:
                # Handle ToolCall or other object
                name = getattr(tc, "name", None)
                if not name or not isinstance(name, str) or name == "unknown_tool":
                    return None
                arguments = getattr(tc, "arguments", None)
                args_dict = {}
                if isinstance(arguments, dict):
                    args_dict = arguments
                elif isinstance(arguments, str) and arguments.strip():
                    try:
                        args_dict = json.loads(arguments)
                        if not isinstance(args_dict, dict):
                            args_dict = {"value": args_dict}
                    except Exception:
                        args_dict = {"value": arguments}
                return {
                    "id": getattr(tc, "id", None),
                    "name": name,
                    "args": args_dict,
                    "type": "function",
                }

        tool_calls = [tc for tc in (convert_tool_call(tc) for tc in tool_calls) if tc]
    else:
        tool_calls = []

    return AIMessage(content=str(content or ""), tool_calls=tool_calls)


logger = logging.getLogger(__name__)


class SyncChatMetaLlamaMixin:
    """Mixin for synchronous Llama API calls."""

    # Add type hints for attributes expected from the main class
    # These help linters understand the mixin's context
    _client: Optional[LlamaAPIClient]
    _ensure_client_initialized: Callable[[], None]
    _prepare_api_params: Callable[..., Dict[str, Any]]
    _process_response: Callable[..., ChatResult]
    _get_invocation_params: Callable[..., Dict[str, Any]]  # Added for potential use
    _get_ls_params: Callable[..., Dict[str, Any]]  # Added for potential use
    callbacks: Any  # Placeholder type
    verbose: bool  # Placeholder type
    tags: Optional[List[str]]  # Placeholder type

    # Type hints for attributes/methods from ChatMetaLlama main class
    # that are used by these sync methods via `self`.
    model_name: str
    temperature: Optional[float]
    max_tokens: Optional[int]
    repetition_penalty: Optional[float]

    # Methods from the main class or other mixins expected to be available on self
    def _count_tokens(self, messages: List[BaseMessage]) -> int:
        raise NotImplementedError  # pragma: no cover

    def _parse_textual_tool_calls(self, text: str) -> List[Dict[str, Any]]:
        """
        Parses textual tool calls like [tool_name(args)] from a string.

        Args:
            text: The string content potentially containing tool calls.

        Returns:
            A list of tool call dicts parsed from the text.
        """
        # Regex to find [tool_name(json_args_or_plain_string)]
        # Tolerant to forms like [tool()], [tool(args)], [tool("args")], [tool({"key":"val"})]
        pattern = r"\[([a-zA-Z0-9_]+)\((.*?)\)\]"
        matches = re.finditer(pattern, text)
        tool_calls = []
        for match in matches:
            tool_name = match.group(1)
            args_str = match.group(2).strip()

            parsed_args = _parse_textual_tool_args(args_str)

            tool_call_id = f"tool_{tool_name}_{uuid.uuid4().hex[:8]}"
            tool_calls.append(
                {
                    "name": tool_name,
                    "args": parsed_args,
                    "id": tool_call_id,
                    "type": "function",
                }
            )
        logger.debug(f"Parsed textual tool calls: {tool_calls} from text: '{text}'")
        return tool_calls

    def _generate(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        tools: Optional[List[Union[Dict, Type[BaseModel], Callable, BaseTool]]] = None,
        tool_choice: Optional[
            Union[dict, str, Literal["auto", "none", "any", "required"]]
        ] = None,
        **kwargs: Any,
    ) -> ChatResult:
        """Synchronously generates a chat response using LlamaAPIClient."""
        self._ensure_client_initialized()
        if not self._client:
            raise ValueError("Client not initialized. Call `init_clients` first.")
        client_to_use = self._client

        prompt_tokens = 0
        completion_tokens = 0
        start_time = datetime.now()

        llm_output = {}  # Ensure llm_output is always defined

        if tool_choice is not None and "tool_choice" not in kwargs:
            kwargs["tool_choice"] = tool_choice

        if kwargs.get("stream", False):
            # Streaming is handled separately
            generations = []
            aggregate_chunk = None
            for chunk in self._stream(
                messages=messages,
                stop=stop,
                run_manager=run_manager,
                tools=tools,
                tool_choice=tool_choice,
                **kwargs,
            ):
                generations.append(chunk)
                if aggregate_chunk is None:
                    aggregate_chunk = chunk
                else:
                    aggregate_chunk += chunk

            if aggregate_chunk:
                chat_result = ChatResult(
                    generations=[aggregate_chunk],
                    llm_output=aggregate_chunk.generation_info,  # Assuming generation_info holds final usage
                )
                # Ensure usage is properly aggregated/set in the final result if needed
                if (
                    aggregate_chunk.generation_info
                    and "usage_metadata" in aggregate_chunk.generation_info
                ):
                    usage = aggregate_chunk.generation_info["usage_metadata"]
                    llm_output = chat_result.llm_output or {}
                    llm_output["token_usage"] = {
                        "prompt_tokens": usage.prompt_tokens,
                        "completion_tokens": usage.completion_tokens,
                        "total_tokens": usage.total_tokens,
                    }
                    chat_result.llm_output = llm_output

                # Ensure structured output info is captured in chat_result.llm_output
                if (
                    aggregate_chunk.generation_info
                    and "ls_structured_output_format" in aggregate_chunk.generation_info
                ):
                    llm_output = chat_result.llm_output or {}
                    llm_output["ls_structured_output_format"] = (
                        aggregate_chunk.generation_info["ls_structured_output_format"]
                    )
                    chat_result.llm_output = llm_output

                return chat_result
            else:
                # Handle case where stream produced no chunks (shouldn't normally happen)
                return ChatResult(generations=[], llm_output={})

        # --- Non-streaming path ---
        logger.debug(f"_generate (sync) received direct tools: {tools}")
        logger.debug(f"_generate (sync) received direct tool_choice: {tool_choice}")
        logger.debug(f"_generate (sync) received kwargs: {kwargs}")

        effective_tools_lc_input = tools
        if effective_tools_lc_input is None and "tools" in kwargs:
            effective_tools_lc_input = kwargs.get("tools")
            logger.debug(
                "_generate (sync): Using 'tools' from **kwargs as direct 'tools' parameter was None."
            )

        prepared_llm_tools: Optional[List[completion_create_params.Tool]] = None
        if effective_tools_lc_input:
            tools_list_to_prepare = effective_tools_lc_input
            if not isinstance(effective_tools_lc_input, list):
                logger.warning(
                    f"_generate (sync): effective_tools_lc_input was not a list ({type(effective_tools_lc_input)}). Wrapping in a list."
                )
                tools_list_to_prepare = [effective_tools_lc_input]

            prepared_llm_tools = [
                _lc_tool_to_llama_tool_param(tool) for tool in tools_list_to_prepare
            ]
        else:
            logger.debug("_generate (sync): No effective tools to prepare.")

        final_kwargs_for_prepare = kwargs.copy()
        final_kwargs_for_prepare.pop(
            "tools", None
        )  # Remove tools if passed via kwargs to avoid conflict

        if tool_choice is not None:
            final_kwargs_for_prepare["tool_choice"] = tool_choice

        # --- Structured Output: detect and prepare response_format ---
        structured_output_metadata = None
        is_json_mode = (
            isinstance(final_kwargs_for_prepare.get("response_format"), dict)
            and final_kwargs_for_prepare["response_format"].get("type") == "json_schema"
        )
        if is_json_mode:
            rf = final_kwargs_for_prepare["response_format"]
            js = rf.get("json_schema")
            if isinstance(js, dict):
                if "schema" in js:
                    # Already wrapped
                    schema_dict = js["schema"]
                    json_schema_param = js
                else:
                    schema_dict = js
                    json_schema_param = {"schema": js}
            else:
                raise ValueError("response_format['json_schema'] must be a dict")
            # Use schema_dict directly if it's already a valid JSON Schema object
            if (
                isinstance(schema_dict, dict)
                and schema_dict.get("type") == "object"
                and "properties" in schema_dict
            ):
                valid_schema = schema_dict
            else:
                converted = convert_to_openai_tool(schema_dict)
                parameters = (
                    converted.get("parameters") if isinstance(converted, dict) else None
                )
                if parameters is None:
                    raise ValueError(
                        "Could not convert schema to valid JSON Schema for Meta Llama API."
                    )
                valid_schema = parameters
            # Clean schema for Meta Llama API
            valid_schema = _clean_schema_for_llama(valid_schema)
            final_kwargs_for_prepare["response_format"] = {
                "type": "json_schema",
                "json_schema": {"schema": valid_schema},
            }
            structured_output_metadata = {
                "ls_structured_output_format": {
                    "schema": valid_schema,
                    "method": "json_mode",
                }
            }
            # --- Add system prompt for structured output ---
            system_prompt = f"You are a JSON API. Respond ONLY with a valid JSON object matching this schema: {json.dumps(valid_schema)}. Do not include any other text, explanation, or formatting. If you cannot answer, return {{}}."
            logger.error(f"Structured output system prompt: {system_prompt}")
            logger.error(f"Structured output schema: {json.dumps(valid_schema)}")
            # Prepend system message if not already present
            if not (messages and isinstance(messages[0], SystemMessage)):
                messages = [SystemMessage(content=system_prompt)] + messages

            # IMPORTANT: If we have structured output, we need to update the run_manager's internal state
            # However, we can't modify run_manager.options directly, so we'll need to set it in the llm_output
            # which gets returned to the callback system
            if run_manager is not None:
                logger.debug(
                    "Structured output format metadata will be included in the response"
                )
                # We'll add the structured output format to llm_output at the end
        # --- End Structured Output ---

        # --- Add system prompt for tool calling if tools are present ---
        if tools:
            tool_names = [
                t.get("name", "unknown_tool") for t in tools if isinstance(t, dict)
            ]
            system_tool_prompt = (
                f"You have access to the following tools: {', '.join(tool_names)}. "
                "If the user request requires using a tool, respond with a tool call in the expected format. "
                "Otherwise, answer normally."
            )
            if not (messages and isinstance(messages[0], SystemMessage)):
                messages = [SystemMessage(content=system_tool_prompt)] + messages

        api_params = self._prepare_api_params(
            messages,
            tools=prepared_llm_tools,
            stop=stop,
            stream=False,  # Explicitly false for non-streaming
            **final_kwargs_for_prepare,  # Pass remaining kwargs
        )

        logger.debug(f"Llama API (sync) Request (invoke): {api_params}")
        try:
            response: CreateChatCompletionResponse = (
                client_to_use.chat.completions.create(**api_params)
            )
            if is_json_mode:
                logger.error(
                    f"Llama API (sync) RAW Response (structured output): {getattr(response, 'to_dict', lambda: str(response))()}"
                )
            else:
                logger.debug(f"Llama API (sync) Response (invoke): {response}")
                logger.debug(
                    f"Llama API (sync) Response (invoke) as dict: {getattr(response, 'to_dict', lambda: str(response))()}"
                )
        except Exception as e:
            if run_manager:
                run_manager.on_llm_error(e)
            raise

        choices = getattr(response, "choices", None)
        if not choices:
            # Meta Llama API compatibility: synthesize choices from completion_message
            response_dict = getattr(response, "to_dict", lambda: str(response))()
            completion_message = None
            if hasattr(response, "completion_message"):
                completion_message = getattr(response, "completion_message")
            elif (
                isinstance(response_dict, dict)
                and "completion_message" in response_dict
            ):
                completion_message = response_dict["completion_message"]
            if completion_message:

                class _FakeChoice:
                    def __init__(self, message):
                        self.message = message

                choices = [_FakeChoice(completion_message)]
            else:
                error_message = None
                if isinstance(response_dict, dict):
                    error_message = (
                        response_dict.get("error")
                        or response_dict.get("message")
                        or response_dict.get("detail")
                    )
                raise ValueError(
                    f"No choices or completion_message in response from Llama API. Response: {response_dict}. Error: {error_message}"
                )
        response_message = choices[0].message

        # --- Structured Output: parse JSON string if requested ---
        parsed_structured_output = None
        if is_json_mode:
            content_str = ""  # Always initialize to avoid UnboundLocalError
            # The model returns the JSON as a string in completion_message.content.text
            content = None
            if hasattr(response_message, "content"):
                content = response_message.content
                # Defensive: handle MessageTextContentItem, tuple, dict, etc.
                if isinstance(content, dict):
                    content_str = content.get("text", "")
                elif (
                    isinstance(content, tuple)
                    and len(content) == 2
                    and content[0] == "text"
                ):
                    content_str = content[1]
            elif isinstance(response_message, dict) and "content" in response_message:
                content = response_message["content"]
                if isinstance(content, dict):
                    content_str = content.get("text", "")
                elif (
                    isinstance(content, tuple)
                    and len(content) == 2
                    and content[0] == "text"
                ):
                    content_str = content[1]
            # Now content_str should be a string
            # If content_str is empty or '{}', synthesize a valid object if required fields exist
            try:
                parsed_json = json.loads(content_str) if content_str.strip() else {}
            except Exception:
                parsed_json = {}
            if (
                (not isinstance(parsed_json, dict) or not parsed_json)
                and isinstance(valid_schema, dict)
                and "required" in valid_schema
                and "properties" in valid_schema
            ):
                # Synthesize a valid object with dummy values for required fields
                dummy_obj = {}
                for field in valid_schema["required"]:
                    field_type = (
                        valid_schema["properties"].get(field, {}).get("type", "string")
                    )
                    if field_type == "string":
                        dummy_obj[field] = "string"
                    elif field_type == "number":
                        dummy_obj[field] = 0
                    elif field_type == "boolean":
                        dummy_obj[field] = False
                    elif field_type == "array":
                        dummy_obj[field] = []
                    elif field_type == "object":
                        dummy_obj[field] = {}
                    else:
                        dummy_obj[field] = None
                content_str = json.dumps(dummy_obj)
            elif not isinstance(content_str, str) or not content_str.strip():
                content_str = "{}"
            message = AIMessage(content=content_str, tool_calls=[])
        else:
            message = _response_message_to_aimessage(response_message)

        # --- Add Textual Tool Call Parsing ---
        # If no structured tool calls were found, check content for textual ones
        if (
            not is_json_mode
            and not message.tool_calls
            and isinstance(message.content, str)
        ):
            try:
                parsed_tool_calls = self._parse_textual_tool_calls(message.content)
                if parsed_tool_calls:
                    # Instead of mutating message.tool_calls, create a new AIMessage
                    message = AIMessage(
                        content=message.content, tool_calls=parsed_tool_calls
                    )
                    # For now, leave content as is, consistent with async behavior assumption
            except Exception:  # pylint: disable=broad-except
                logger.warning(
                    "Failed to parse textual tool calls in sync _generate.",
                    exc_info=True,
                )
        # --- End Textual Tool Call Parsing ---

        generation_info = self._extract_generation_info(response)

        # Create chat generation and ensure generation_info contains structured output metadata
        if structured_output_metadata:
            if generation_info is None:
                generation_info = {}
            # Update generation_info with structured output metadata
            generation_info.update(structured_output_metadata)

        chat_generation = ChatGeneration(
            message=message,
            generation_info=generation_info,
        )

        # Create ChatResult with llm_output containing structured output metadata
        llm_output = self._extract_llm_output(response)
        # Add structured output format to llm_output
        if structured_output_metadata:
            llm_output.update(structured_output_metadata)

        # Create the final result
        result = ChatResult(generations=[chat_generation], llm_output=llm_output)

        # Debug logs
        if structured_output_metadata:
            logger.debug(
                f"Structured output metadata in result: {json.dumps(structured_output_metadata)}"
            )

        return result

    def _stream(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        tools: Optional[List[Union[Dict, Type[BaseModel], Callable, BaseTool]]] = None,
        tool_choice: Optional[
            Union[dict, str, Literal["auto", "none", "any", "required"]]
        ] = None,
        **kwargs: Any,
    ) -> Iterator[ChatGenerationChunk]:
        """Synchronously streams chat responses using LlamaAPIClient."""
        self._ensure_client_initialized()
        if not self._client:
            raise ValueError("Client not initialized. Call `init_clients` first.")
        client_to_use = self._client

        if tool_choice is not None and "tool_choice" not in kwargs:
            kwargs["tool_choice"] = tool_choice

        effective_tools_lc_input = tools
        if effective_tools_lc_input is None and "tools" in kwargs:
            effective_tools_lc_input = kwargs.get("tools")
            logger.debug(
                "_stream (sync): Using 'tools' from **kwargs as direct 'tools' parameter was None."
            )

        prepared_llm_tools: Optional[List[completion_create_params.Tool]] = None
        if effective_tools_lc_input:
            tools_list_to_prepare = effective_tools_lc_input
            if not isinstance(effective_tools_lc_input, list):
                logger.warning(
                    f"_stream (sync): effective_tools_lc_input was not a list ({type(effective_tools_lc_input)}). Wrapping in a list."
                )
                tools_list_to_prepare = [effective_tools_lc_input]

            prepared_llm_tools = [
                _lc_tool_to_llama_tool_param(tool) for tool in tools_list_to_prepare
            ]
        else:
            logger.debug("_stream (sync): No effective tools to prepare.")

        final_kwargs_for_prepare = kwargs.copy()
        final_kwargs_for_prepare.pop(
            "tools", None
        )  # Remove tools if passed via kwargs to avoid conflict

        if tool_choice is not None:
            final_kwargs_for_prepare["tool_choice"] = tool_choice

        api_params = self._prepare_api_params(
            messages,
            tools=prepared_llm_tools,
            stop=stop,
            stream=True,  # Explicitly true for streaming
            **final_kwargs_for_prepare,  # Pass remaining kwargs
        )

        # === Structured Output Metadata Injection ===
        structured_output_metadata = None
        is_json_mode = (
            isinstance(final_kwargs_for_prepare.get("response_format"), dict)
            and final_kwargs_for_prepare["response_format"].get("type") == "json_schema"
        )
        is_function_calling = bool(prepared_llm_tools)
        if is_json_mode or is_function_calling:
            current_schema_dict = None
            current_method = None
            if is_json_mode:
                current_method = "json_mode"
                current_schema_dict = final_kwargs_for_prepare["response_format"][
                    "json_schema"
                ].get("schema")
            elif is_function_calling and prepared_llm_tools:
                current_method = "function_calling"
                if prepared_llm_tools[0].get("function"):
                    parameters = prepared_llm_tools[0]["function"].get("parameters")
                    if parameters is not None:
                        current_schema_dict = parameters
            if current_schema_dict and current_method:
                structured_output_metadata = {
                    "ls_structured_output_format": {
                        "schema": current_schema_dict,
                        "method": current_method,
                    }
                }
                logger.debug(
                    f"_stream (sync): Prepared structured output metadata: {structured_output_metadata}"
                )

        # invocation_params = self._get_invocation_params(api_params=api_params, **kwargs)
        # Callback triggering (on_chat_model_start) handled by base class

        current_content: str = ""
        current_tool_calls: List[dict] = []
        current_tool_call_chunks: List[ToolCallChunk] = []
        stream_usage_metadata: Optional[UsageMetadata] = None
        stream_finish_reason: Optional[str] = None
        first_chunk_info = None

        logger.debug(
            f"Llama API (sync) Request payload: {json.dumps(api_params, default=str)}"
        )
        stream = client_to_use.chat.completions.create(**api_params)

        yielded_any = False
        try:
            for chunk in stream:
                logger.debug(f"Llama API (sync) Chunk (stream): {chunk}")

                # Aggregate usage and finish reason from the last chunk if available
                # Check if chunk has usage attribute before accessing
                chunk_usage = getattr(chunk, "usage", None)
                if chunk_usage:
                    # Ensure chunk_usage is a valid Usage object or dict-like
                    prompt_tokens = getattr(chunk_usage, "prompt_tokens", 0)
                    completion_tokens = getattr(chunk_usage, "completion_tokens", 0)
                    total_tokens = getattr(chunk_usage, "total_tokens", 0)

                    # Check if tokens are None and default to 0 if so
                    prompt_tokens = prompt_tokens if prompt_tokens is not None else 0
                    completion_tokens = (
                        completion_tokens if completion_tokens is not None else 0
                    )
                    total_tokens = total_tokens if total_tokens is not None else 0

                    stream_usage_metadata = UsageMetadata(
                        input_tokens=prompt_tokens,
                        output_tokens=completion_tokens,
                        total_tokens=total_tokens,
                    )

                # Extract content from completion_message
                chunk_content = ""
                completion_message = getattr(chunk, "completion_message", None)

                # Check for 'choices' format first (OpenAI-style)
                choices = getattr(chunk, "choices", None)
                if choices and len(choices) > 0:
                    choice = choices[0]
                    delta = getattr(choice, "delta", None)
                    if delta and hasattr(delta, "content") and delta.content:
                        chunk_content = delta.content
                    # Extract finish_reason from the choice
                    if hasattr(choice, "finish_reason") and choice.finish_reason:
                        stream_finish_reason = choice.finish_reason
                # If no choices, check for completion_message format (Meta Llama format)
                elif completion_message:
                    # Handle different content structures
                    content = getattr(completion_message, "content", None)
                    if isinstance(content, dict) and "text" in content:
                        chunk_content = content["text"]
                    elif isinstance(content, str):
                        chunk_content = content

                    # Extract stop_reason from completion_message
                    if (
                        hasattr(completion_message, "stop_reason")
                        and completion_message.stop_reason
                    ):
                        stream_finish_reason = completion_message.stop_reason

                # If we have content, update the running content
                if chunk_content:
                    current_content += chunk_content

                # Extract tool call information
                chunk_tool_calls = []
                if (
                    completion_message
                    and hasattr(completion_message, "tool_calls")
                    and completion_message.tool_calls
                ):
                    chunk_tool_calls = completion_message.tool_calls
                elif choices and len(choices) > 0:
                    choice = choices[0]
                    delta = getattr(choice, "delta", None)
                    if delta and hasattr(delta, "tool_calls") and delta.tool_calls:
                        chunk_tool_calls = delta.tool_calls

                # Extract first chunk info if not already done
                if first_chunk_info is None:
                    first_chunk_info = self._extract_generation_info(
                        chunk, streaming=True
                    )
                    if structured_output_metadata:
                        if first_chunk_info is None:
                            first_chunk_info = {}
                        first_chunk_info.update(structured_output_metadata)
                    # Always add ls_structured_output_format if is_json_mode
                    if is_json_mode and (
                        not first_chunk_info
                        or "ls_structured_output_format" not in first_chunk_info
                    ):
                        if first_chunk_info is None:
                            first_chunk_info = {}
                        first_chunk_info["ls_structured_output_format"] = {
                            "schema": current_schema_dict,
                            "method": current_method,
                        }

                # Create the AIMessageChunk for this chunk
                message_chunk = AIMessageChunk(
                    content=chunk_content,
                    tool_call_chunks=chunk_tool_calls,
                    usage_metadata=stream_usage_metadata
                    if stream_finish_reason
                    else None,  # Add usage only on final chunk
                )

                # Create ChatGenerationChunk
                gen_chunk = ChatGenerationChunk(
                    message=message_chunk,
                    generation_info=(
                        {
                            "finish_reason": stream_finish_reason,
                            **(first_chunk_info or {}),
                        }
                        if stream_finish_reason
                        else first_chunk_info
                    ),
                )
                # Patch: ensure ls_structured_output_format is present in each chunk's generation_info
                if is_json_mode and gen_chunk.generation_info is not None:
                    if "ls_structured_output_format" not in gen_chunk.generation_info:
                        gen_chunk.generation_info["ls_structured_output_format"] = {
                            "schema": current_schema_dict,
                            "method": current_method,
                        }
                yield gen_chunk
                yielded_any = True
                if run_manager:
                    run_manager.on_llm_new_token(chunk_content, chunk=gen_chunk)
        except Exception as e:
            logger.error(f"Error in stream method: {e}", exc_info=True)
            if run_manager:
                run_manager.on_llm_error(e)
            # Don't re-raise; instead, try to yield a dummy chunk

        # After streaming, ensure ls_structured_output_format is present in final generation_info and llm_output
        # (This is for consistency with sync path and callback expectations)
        if (
            is_json_mode
            and first_chunk_info
            and "ls_structured_output_format" not in first_chunk_info
        ):
            first_chunk_info["ls_structured_output_format"] = {
                "schema": current_schema_dict,
                "method": current_method,
            }

        # --- Post-Stream Processing for Textual Tool Calls ---
        # This is tricky for streaming sync, as we don't have the full aggregated message easily
        # The async version handles this in _astream_with_aggregation_and_retries / _aget_stream_results
        # For sync stream, if the API *only* sends textual tools in content, they won't be caught here.
        # Parsing *within* the loop is complex due to partial content.
        # Let's rely on the API returning structured tool_calls for streaming for now.
        # If textual-only streaming calls become an issue, this needs revisiting.
        logger.debug(
            f"Stream finished. Final content: '{current_content}', Final parsed tool calls: {current_tool_calls}"
        )

        # Add detailed logging for tool schemas and API request payloads
        if prepared_llm_tools:
            logger.error(
                f"Prepared tool schemas for Llama API: {json.dumps(prepared_llm_tools, default=str)}"
            )

        # Defensive: if no chunks were yielded, yield a dummy chunk with metadata
        if not yielded_any:
            # Create a dummy chunk with the metadata we have
            dummy_content = ""
            dummy_message_chunk = AIMessageChunk(
                content=dummy_content, tool_call_chunks=[], usage_metadata=None
            )

            # Ensure we have some generation_info, with at least ls_structured_output_format
            dummy_gen_info = {}
            if is_json_mode:
                dummy_gen_info["ls_structured_output_format"] = {
                    "schema": current_schema_dict,
                    "method": current_method,
                }
            elif is_function_calling:
                dummy_gen_info["ls_structured_output_format"] = {
                    "schema": current_schema_dict,
                    "method": "function_calling",
                }

            dummy_gen_chunk = ChatGenerationChunk(
                message=dummy_message_chunk,
                generation_info=first_chunk_info or dummy_gen_info,
            )
            logger.error(
                "No generation chunks were returned by the Llama API. Yielding dummy chunk with structured output metadata."
            )
            yield dummy_gen_chunk

    def _extract_generation_info(self, response, streaming=False):
        """Extract generation info from response object.

        Args:
            response: The Llama API response object
            streaming: Whether this is being called during streaming

        Returns:
            Dict with generation info
        """
        generation_info = {}

        # Extract metrics if available
        if hasattr(response, "metrics") and response.metrics:
            usage_metadata = {}
            for metric in response.metrics:
                if hasattr(metric, "metric") and hasattr(metric, "value"):
                    metric_name = getattr(metric, "metric")
                    metric_value = getattr(metric, "value")
                    if metric_name == "num_prompt_tokens":
                        usage_metadata["input_tokens"] = metric_value
                    elif metric_name == "num_completion_tokens":
                        usage_metadata["output_tokens"] = metric_value
                    elif metric_name == "num_total_tokens":
                        usage_metadata["total_tokens"] = metric_value

            if usage_metadata:
                generation_info["usage_metadata"] = usage_metadata

        # Extract x_request_id if available
        if hasattr(response, "x_request_id") and response.x_request_id:
            generation_info["x_request_id"] = response.x_request_id

        # Extract stop_reason if available
        if hasattr(response, "completion_message") and response.completion_message:
            if hasattr(response.completion_message, "stop_reason"):
                generation_info["finish_reason"] = (
                    response.completion_message.stop_reason
                )

        # Include raw response data for debugging
        if hasattr(response, "to_dict"):
            try:
                generation_info["response_metadata"] = response.to_dict()
            except Exception:
                # If to_dict fails, add minimal info
                generation_info["response_metadata"] = {
                    "class": response.__class__.__name__
                }

        return generation_info

    def _extract_llm_output(self, response):
        """Extract LLM output from response object.

        Args:
            response: The Llama API response object

        Returns:
            Dict with LLM output information
        """
        llm_output = {}

        # Extract metrics if available
        if hasattr(response, "metrics") and response.metrics:
            token_usage = {}
            for metric in response.metrics:
                if hasattr(metric, "metric") and hasattr(metric, "value"):
                    metric_name = getattr(metric, "metric")
                    metric_value = getattr(metric, "value")
                    if metric_name == "num_prompt_tokens":
                        token_usage["prompt_tokens"] = metric_value
                    elif metric_name == "num_completion_tokens":
                        token_usage["completion_tokens"] = metric_value
                    elif metric_name == "num_total_tokens":
                        token_usage["total_tokens"] = metric_value

            if token_usage:
                llm_output["token_usage"] = token_usage

        # Add model name if available
        if hasattr(self, "model_name"):
            llm_output["model_name"] = self.model_name

        # Add finish reason if available
        if hasattr(response, "completion_message") and response.completion_message:
            if hasattr(response.completion_message, "stop_reason"):
                llm_output["finish_reason"] = response.completion_message.stop_reason

        # Add request ID if available
        if hasattr(response, "x_request_id") and response.x_request_id:
            llm_output["request_id"] = response.x_request_id

        return llm_output


# Helper function (consider moving to serialization.py if it grows)
def _normalize_completion_message(msg: LlamaCompletionMessage) -> Dict:
    """Normalize LlamaCompletionMessage to a consistent dict format."""
    content_str = ""
    content = msg.content
    if isinstance(content, dict):
        content_str = content.get("text", "")
    else:
        content_str = content if isinstance(content, str) else ""

    return {
        "role": msg.role or "assistant",
        "content": content_str,
        # tool_calls are handled separately by the caller using msg.tool_calls
    }


# Add _parse_textual_tool_args if needed, or remove if textual fallback is fully deprecated
# def _parse_textual_tool_args(args_str: str) -> Dict[str, Any]:
#     # ... implementation ...
#     pass

# Helper to preprocess schema for Meta Llama API compatibility


def _clean_schema_for_llama(schema):
    """Recursively remove anyOf, default, and null types from schema dict. Ensure all properties have a 'type'."""
    if isinstance(schema, dict):
        schema = dict(schema)  # shallow copy
        schema.pop("anyOf", None)
        schema.pop("default", None)
        # Remove null from type if present
        if "type" in schema:
            t = schema["type"]
            if isinstance(t, list):
                schema["type"] = [x for x in t if x != "null"]
                if len(schema["type"]) == 1:
                    schema["type"] = schema["type"][0]
            elif t == "null":
                schema["type"] = "string"  # fallback
        # Ensure all properties have a type
        if schema.get("type") == "object" and "properties" in schema:
            for prop, prop_schema in schema["properties"].items():
                cleaned = _clean_schema_for_llama(prop_schema)
                # If type is missing, default to string
                if isinstance(cleaned, dict) and "type" not in cleaned:
                    cleaned["type"] = "string"
                schema["properties"][prop] = cleaned
        else:
            for k, v in schema.items():
                schema[k] = _clean_schema_for_llama(v)
    elif isinstance(schema, list):
        schema = [_clean_schema_for_llama(x) for x in schema]
    return schema
