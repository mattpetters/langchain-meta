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
from llama_api_client.types.chat import (
    completion_create_params,
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

logger = logging.getLogger(__name__)


class SyncChatMetaLlamaMixin:
    """Mixin class to hold synchronous methods for ChatMetaLlama."""

    # Type hints for attributes/methods from ChatMetaLlama main class
    # that are used by these sync methods via `self`.
    _client: Optional[LlamaAPIClient]
    model_name: str
    temperature: Optional[float]
    max_tokens: Optional[int]
    repetition_penalty: Optional[float]

    # Methods from the main class or other mixins expected to be available on self
    def _ensure_client_initialized(self) -> None:
        raise NotImplementedError  # pragma: no cover

    def _prepare_api_params(
        self,
        messages: List[BaseMessage],
        tools: Optional[List[Any]] = None,
        stop: Optional[List[str]] = None,
        stream: bool = False,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        raise NotImplementedError  # pragma: no cover

    def _count_tokens(self, messages: List[BaseMessage]) -> int:
        raise NotImplementedError  # pragma: no cover

    def _get_invocation_params(self, **kwargs: Any) -> Dict[str, Any]:
        raise NotImplementedError  # pragma: no cover

    # _lc_tool_to_llama_tool_param is imported and used directly
    # _lc_message_to_llama_message_param is imported and used by _prepare_api_params (assumed to be on self or accessible)

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
        """Generate a chat response using the sync API client."""
        self._ensure_client_initialized()
        if self._client is None:
            raise ValueError("LlamaAPIClient not initialized.")

        active_client = kwargs.get("client") or self._client
        if not active_client:
            raise ValueError("Could not obtain an active LlamaAPIClient.")

        start_time = datetime.now()
        input_tokens = self._count_tokens(messages)

        # === Callback Handling Start ===
        llm_run_manager: Optional[CallbackManagerForLLMRun] = None
        if run_manager:
            # Check if run_manager is already the child LLM manager or needs get_child()
            if isinstance(run_manager, CallbackManagerForLLMRun):
                llm_run_manager = run_manager  # It's already the child
                logger.debug(
                    "Inside _generate: run_manager is already CallbackManagerForLLMRun."
                )
            elif hasattr(run_manager, "get_child"):
                llm_run_manager = run_manager.get_child()  # Get child manager
                logger.debug("Inside _generate: Called run_manager.get_child().")
            else:
                logger.warning(
                    f"Inside _generate: run_manager is of unexpected type {type(run_manager)} and has no get_child. Callbacks may not work correctly."
                )
                # Attempt to use it directly, hoping it has the necessary methods.
                # This branch might need further refinement based on observed types.
                # For now, we assume if it's not CallbackManagerForLLMRun and doesn't have get_child,
                # it might be a custom manager that should be used directly.
                # However, this is less common for standard LangChain flows.
                # A more robust solution might involve stricter type checking or specific handling
                # for known alternative manager types if they exist.
                llm_run_manager = run_manager  # Fallback, hoping for the best

        # === Callback Handling End ===

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
            typed_tools_list_to_prepare: List[
                Union[Dict, Type[BaseModel], Callable, BaseTool]
            ] = tools_list_to_prepare

            prepared_llm_tools = [
                _lc_tool_to_llama_tool_param(tool)
                for tool in typed_tools_list_to_prepare
                # Add a defensive check here, although the real fix might be needed in _lc_tool_to_llama_tool_param
                if isinstance(tool, dict)
                and tool.get("function")
                and isinstance(tool["function"], dict)  # Basic structure check
            ]
        else:
            logger.debug("_generate (sync): No effective tools to prepare.")

        # Determine structured output details for metadata
        structured_output_metadata = None
        is_json_mode = (
            isinstance(kwargs.get("response_format"), dict)
            and kwargs["response_format"].get("type") == "json_schema"
        )
        is_function_calling = bool(prepared_llm_tools)
        if is_json_mode or is_function_calling:
            current_schema_dict = None
            current_method = None
            if is_json_mode:
                current_method = "json_mode"
                current_schema_dict = kwargs["response_format"]["json_schema"].get(
                    "schema"
                )
            elif is_function_calling and prepared_llm_tools:
                current_method = "function_calling"
                if prepared_llm_tools[0].get("function") and prepared_llm_tools[0][
                    "function"
                ].get("parameters"):
                    current_schema_dict = prepared_llm_tools[0]["function"][
                        "parameters"
                    ]
            if current_schema_dict and current_method:
                structured_output_metadata = {
                    "ls_structured_output_format": {
                        "schema": current_schema_dict,
                        "method": current_method,
                    }
                }
                logger.debug(
                    f"_generate: Prepared structured output metadata: {structured_output_metadata}"
                )

        final_kwargs_for_prepare = kwargs.copy()
        final_kwargs_for_prepare.pop("tools", None)
        if tool_choice is not None:
            final_kwargs_for_prepare["tool_choice"] = tool_choice

        api_params = self._prepare_api_params(
            messages=messages,
            tools=prepared_llm_tools,
            stop=stop,
            stream=False,
            **final_kwargs_for_prepare,
        )

        logger.debug(f"Llama API (sync) Request: {api_params}")
        try:
            call_result = active_client.chat.completions.create(**api_params)
            logger.debug(f"Llama API (sync) Response: {call_result}")
        except Exception as e:
            if llm_run_manager:  # Check if llm_run_manager was successfully obtained
                llm_run_manager.on_llm_error(error=e)  # type: ignore[attr-defined]
            raise e

        result_msg = (
            call_result.completion_message
            if hasattr(call_result, "completion_message")
            else None
        )
        content_str = ""

        # Enhanced content extraction for improved reliability
        if result_msg:
            # Direct attribute access method - attempt 1
            if hasattr(result_msg, "content"):
                content = getattr(result_msg, "content")
                if isinstance(content, dict) and "text" in content:
                    content_str = content["text"]
                elif isinstance(content, str):
                    content_str = content

            # If the above didn't work, try dictionary-based access - attempt 2
            if not content_str and hasattr(result_msg, "to_dict"):
                try:
                    result_dict = result_msg.to_dict()
                    if isinstance(result_dict, dict) and "content" in result_dict:
                        content_dict = result_dict["content"]
                        if isinstance(content_dict, dict) and "text" in content_dict:
                            content_str = content_dict["text"]
                        elif isinstance(content_dict, str):
                            content_str = content_dict
                except (AttributeError, TypeError, KeyError):
                    pass

        # If still no content but we have response data, traverse known structures - attempt 3
        if not content_str and hasattr(call_result, "to_dict"):
            try:
                full_result = call_result.to_dict()
                if isinstance(full_result, dict):
                    # Try to extract from completion_message
                    if "completion_message" in full_result:
                        comp_msg = full_result["completion_message"]
                        if isinstance(comp_msg, dict) and "content" in comp_msg:
                            content = comp_msg["content"]
                            if isinstance(content, dict) and "text" in content:
                                content_str = content["text"]
                            elif isinstance(content, str):
                                content_str = content

                    # If there's still no content but response_metadata exists and has completion_message
                    if not content_str and "response_metadata" in full_result:
                        response_meta = full_result["response_metadata"]
                        if (
                            isinstance(response_meta, dict)
                            and "completion_message" in response_meta
                        ):
                            comp_msg = response_meta["completion_message"]
                            if isinstance(comp_msg, dict) and "content" in comp_msg:
                                content = comp_msg["content"]
                                if isinstance(content, dict) and "text" in content:
                                    content_str = content["text"]
                                elif isinstance(content, str):
                                    content_str = content
            except (AttributeError, TypeError, KeyError):
                pass

        tool_calls_data: List[Dict] = []
        generation_info: Dict[str, Any] = {}  # Initialize generation_info here

        if result_msg and hasattr(result_msg, "tool_calls") and result_msg.tool_calls:
            processed_tool_calls: List[Dict] = []
            for idx, tc in enumerate(result_msg.tool_calls):
                tc_id = getattr(tc, "id", None)
                if not tc_id:
                    tc_id = f"llama_tc_{idx}"
                if not tc_id:
                    tc_id = str(uuid.uuid4())

                tc_func = tc.function if hasattr(tc, "function") else None
                tc_name = getattr(tc_func, "name", None) if tc_func else None
                tc_args_str = getattr(tc_func, "arguments", "") if tc_func else ""

                if tc_name and not isinstance(tc_name, str):
                    tc_name = (
                        str(tc_name) if hasattr(tc_name, "__str__") else "unknown_tool"
                    )

                try:
                    parsed_args = json.loads(tc_args_str) if tc_args_str else {}
                    final_args = (
                        {"value": str(parsed_args)}
                        if not isinstance(parsed_args, dict)
                        else parsed_args
                    )
                except json.JSONDecodeError:
                    # Try our malformed args parser for cases like 'name="value", key2="value2"'
                    logger.debug(
                        f"JSON parsing failed, trying malformed args parser for: {tc_args_str}"
                    )
                    final_args = parse_malformed_args_string(tc_args_str)
                except Exception as e:
                    logger.warning(
                        f"Unexpected error processing tool call arguments for {tc_name}: {e}. Representing as string."
                    )
                    final_args = {"value": tc_args_str}

                # Defensive: always ensure id, name, args are properly set
                if not tc_id:
                    tc_id = str(uuid.uuid4())
                if not tc_name:
                    tc_name = "unknown_tool"
                if not isinstance(final_args, dict):
                    final_args = {"value": str(final_args)}

                processed_tool_calls.append(
                    {
                        "id": tc_id,
                        "type": "function",
                        "name": tc_name,
                        "args": final_args,
                    }
                )
            tool_calls_data = processed_tool_calls
        elif (
            prepared_llm_tools
            and content_str
            and content_str.startswith("[")
            and content_str.endswith("]")
        ):
            # If no tool_calls from API, try to parse from content_str if tools were provided and content looks like a textual tool call
            logger.debug(
                f"No structured tool_calls from API. Attempting to parse textual tool call from content: {content_str}"
            )
            match = re.fullmatch(
                r"\s*\[\s*([a-zA-Z0-9_]+)\s*(?:\(\s*(.*?)\s*\))?\s*\]\s*", content_str
            )
            if match:
                tool_name_from_content = match.group(1)
                args_str_from_content = match.group(2)
                available_tool_names = [
                    t["function"]["name"]
                    for t in prepared_llm_tools
                    if isinstance(t, dict)
                    and "function" in t
                    and "name" in t["function"]
                ]
                if tool_name_from_content in available_tool_names:
                    logger.info(
                        f"Parsed textual tool call for '{tool_name_from_content}' from content."
                    )
                    tool_call_id = str(uuid.uuid4())
                    parsed_args = {}
                    if args_str_from_content:
                        try:
                            # First try the standard LangChain parser
                            parsed_args = _parse_textual_tool_args(
                                args_str_from_content
                            )
                        except Exception as e:
                            logger.warning(
                                f"Failed to parse arguments '{args_str_from_content}' for textual tool call '{tool_name_from_content}': {e}. Trying fallback parser."
                            )
                            # Use our fallback parser for malformed argument strings
                            parsed_args = parse_malformed_args_string(
                                args_str_from_content
                            )

                    # Defensive: always ensure all fields are properly set
                    if not tool_call_id:
                        tool_call_id = str(uuid.uuid4())
                    if not tool_name_from_content:
                        tool_name_from_content = "unknown_tool"
                    if not isinstance(parsed_args, dict):
                        parsed_args = {"value": str(parsed_args)}

                    tool_calls_data.append(
                        {
                            "id": tool_call_id,
                            "name": tool_name_from_content,
                            "args": parsed_args,
                            "type": "function",  # LangChain expects this structure
                        }
                    )
                    content_str = (
                        ""  # Clear content as it was a tool call representation
                    )
                    logger.debug(f"Manually constructed tool_calls: {tool_calls_data}")
                    # If we manually created tool_calls, the stop_reason should reflect that.
                    # We'll store this in generation_info, which AIMessage can use.
                    generation_info["finish_reason"] = "tool_calls"
                else:
                    logger.warning(
                        f"Textual tool call '{tool_name_from_content}' found in content, but not in available tools: {available_tool_names}"
                    )
            else:
                logger.debug(
                    f"Content '{content_str}' did not match textual tool call pattern."
                )

        message = AIMessage(
            content=content_str or "",
            tool_calls=tool_calls_data,
            generation_info=generation_info if generation_info else None,
        )
        prompt_tokens = input_tokens  # re-assign from initial count
        completion_tokens = 0

        if result_msg and hasattr(result_msg, "stop_reason") and result_msg.stop_reason:
            generation_info["finish_reason"] = result_msg.stop_reason
        elif hasattr(call_result, "stop_reason") and call_result.stop_reason:
            generation_info["finish_reason"] = call_result.stop_reason

        if (
            hasattr(call_result, "metrics")
            and call_result.metrics
            and isinstance(call_result.metrics, list)
        ):
            usage_meta = {}
            for item in call_result.metrics:
                if hasattr(item, "metric") and hasattr(item, "value"):
                    # Cast value to int here
                    metric_value = int(item.value) if item.value is not None else 0
                    if item.metric == "num_prompt_tokens":
                        usage_meta["input_tokens"] = metric_value
                        prompt_tokens = metric_value
                    elif item.metric == "num_completion_tokens":
                        usage_meta["output_tokens"] = metric_value
                        completion_tokens = metric_value
                    elif item.metric == "num_total_tokens":
                        usage_meta["total_tokens"] = metric_value
            if usage_meta:
                generation_info["usage_metadata"] = usage_meta
                if hasattr(message, "usage_metadata"):  # Check before assigning
                    message.usage_metadata = usage_meta  # type: ignore[assignment]
        elif hasattr(call_result, "usage") and call_result.usage:  # Fallback
            usage_data = call_result.usage
            # Cast values to int here
            prompt_tokens = int(getattr(usage_data, "prompt_tokens", 0))
            completion_tokens = int(getattr(usage_data, "completion_tokens", 0))
            total_tokens = int(getattr(usage_data, "total_tokens", 0))
            usage_meta = {
                "input_tokens": prompt_tokens,
                "output_tokens": completion_tokens,
                "total_tokens": total_tokens,  # Use already casted int values
            }
            # prompt_tokens and completion_tokens are already updated above
            if any(usage_meta.values()):
                generation_info["usage_metadata"] = usage_meta
                if hasattr(message, "usage_metadata"):  # Check before assigning
                    message.usage_metadata = usage_meta  # type: ignore[assignment]

        if hasattr(call_result, "x_request_id") and call_result.x_request_id:
            generation_info["x_request_id"] = call_result.x_request_id
        # generation_info["response_metadata"] = call_result.to_dict() # This would overwrite our potential manual finish_reason
        # Preserve existing generation_info and add to it carefully
        response_metadata_dict = call_result.to_dict()
        if (
            "response_metadata" not in generation_info
        ):  # if we haven't manually set parts of it
            generation_info["response_metadata"] = response_metadata_dict
        else:  # Merge, with our manual values taking precedence if keys conflict (e.g. finish_reason)
            generation_info["response_metadata"] = {
                **response_metadata_dict,
                **generation_info.get("response_metadata", {}),
                **generation_info,
            }
            # The above merge is a bit complex, simplify: ensure original response_metadata is base, then overlay our gen_info
            base_response_meta = response_metadata_dict
            current_gen_info = (
                generation_info.copy()
            )  # our potentially modified generation_info
            generation_info = base_response_meta  # start with full API response
            generation_info.update(current_gen_info)  # overlay our modifications

        end_time = datetime.now()
        duration = (end_time - start_time).total_seconds()
        generation_info["duration"] = duration

        result = ChatResult(
            generations=[
                ChatGeneration(message=message, generation_info=generation_info)
            ]
        )

        # --- Standardize llm_output for callbacks ---
        llm_output_data = {
            "model_name": self.model_name,
            # Ensure token_usage is a dictionary within llm_output
            "token_usage": {
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "total_tokens": prompt_tokens + completion_tokens,
            },
            "system_fingerprint": getattr(
                call_result, "system_fingerprint", None
            ),  # Add if available
            "request_id": getattr(
                call_result, "x_request_id", None
            ),  # Add if available
            "finish_reason": generation_info.get("finish_reason"),  # Add if available
            # Include the raw response if needed for debugging, but maybe exclude from standard callback data
            "raw_response_metadata": call_result.to_dict(),
        }
        result.llm_output = llm_output_data  # Assign the standardized dict
        # --- End Standardization ---

        # === Callback Handling Start for on_llm_end ===
        if llm_run_manager:
            try:
                # The on_llm_end call expects the ChatResult object directly
                # The llm_output within the result object is now standardized
                llm_run_manager.on_llm_end(result)  # type: ignore[attr-defined]
            except Exception as e:
                logger.warning(f"Error in on_llm_end callback: {str(e)}")
                if (
                    isinstance(e, KeyError) and e.args and e.args[0] == 0
                ):  # Check args exist before indexing
                    logger.error(
                        f"(Sync - Still seeing KeyError(0)) Detail: Result llm_output: {result.llm_output}"
                    )

        # === Callback Handling End for on_llm_end ===

        # Trigger on_chat_model_start callback with metadata
        if llm_run_manager:
            invocation_params = self._get_invocation_params(
                api_params=api_params, **final_kwargs_for_prepare
            )
            # This callback should be handled by the base class before _generate is called.
            # If we need to pass options/metadata, it should happen via the config.
            pass  # Placeholder - Callback triggering is handled by base class

        if self.temperature is not None and "temperature" not in api_params:
            api_params["temperature"] = self.temperature

        return result

    def _stream(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> Iterator[ChatGenerationChunk]:
        """Main synchronous streaming method for Llama API."""
        self._ensure_client_initialized()
        if self._client is None:
            raise ValueError("LlamaAPIClient not initialized.")

        active_client = kwargs.get("client") or self._client
        if not active_client:
            raise ValueError("Could not obtain an active LlamaAPIClient.")

        effective_tools_lc_input = kwargs.get("tools")
        # if ( # Commenting out this check as it might be redundant with _prepare_api_params
        #     effective_tools_lc_input is None and "tools" in kwargs
        # ):
        #     effective_tools_lc_input = kwargs.get("tools")
        #     logger.debug(
        #         "_stream (sync): Using 'tools' from **kwargs as direct 'tools' parameter was None."
        #     )

        prepared_llm_tools: Optional[List[completion_create_params.Tool]] = None
        if effective_tools_lc_input:
            tools_list_to_prepare = effective_tools_lc_input
            if not isinstance(effective_tools_lc_input, list):
                # logger.warning( # Reducing log noise
                #     f"_stream (sync): effective_tools_lc_input was not a list ({type(effective_tools_lc_input)}). Wrapping in list."
                # )
                tools_list_to_prepare = [effective_tools_lc_input]

            # Ensure correct type for the list comprehension
            typed_tools_list_to_prepare: List[
                Union[Dict, Type[BaseModel], Callable, BaseTool]
            ] = tools_list_to_prepare

            prepared_llm_tools = [
                _lc_tool_to_llama_tool_param(tool)
                for tool in typed_tools_list_to_prepare
                # Add a defensive check here, although the real fix might be needed in _lc_tool_to_llama_tool_param
                if isinstance(tool, dict)
                and tool.get("function")
                and isinstance(tool["function"], dict)  # Basic structure check
            ]
        # else: # Reducing log noise
        # logger.debug("_stream (sync): No effective tools to prepare.")

        final_kwargs_for_prepare = kwargs.copy()
        final_kwargs_for_prepare.pop(
            "tools", None
        )  # Ensure 'tools' from bind_tools doesn't go directly to _prepare_api_params if it expects the raw LC format

        api_params = self._prepare_api_params(
            messages,
            tools=prepared_llm_tools,  # Pass the Llama-formatted tools
            stop=stop,
            stream=True,
            **final_kwargs_for_prepare,
        )

        all_chunks_for_callback: List[ChatGenerationChunk] = []
        aggregated_tool_calls_buffer: Dict[
            str, Dict[str, Any]
        ] = {}  # tool_id -> {'id', 'name', 'args_str', 'index'}
        next_tool_call_chunk_index = 0  # For LangChain's ToolCallChunk.index

        try:
            # logger.debug(f"Llama API (sync stream) Request: {api_params}") # Reducing log noise
            for chunk in active_client.chat.completions.create(**api_params):
                # logger.debug(f"Llama API (sync stream) Stream Chunk: {chunk.to_dict()}") # Reducing log noise

                chunk_dict = (
                    chunk.to_dict()
                )  # Keep for metadata like x_request_id, usage
                generation_info: Dict[str, Any] = {}
                content_delta_str = ""  # Text content in this specific chunk
                tool_call_chunks_for_lc: List[
                    ToolCallChunk
                ] = []  # ToolCallChunks for this ChatGenerationChunk

                llama_chunk_completion_message = getattr(
                    chunk, "completion_message", None
                )

                if llama_chunk_completion_message:
                    # Handle text content delta
                    if (
                        hasattr(llama_chunk_completion_message, "content")
                        and llama_chunk_completion_message.content
                    ):
                        content_part = llama_chunk_completion_message.content
                        if isinstance(content_part, dict) and "text" in content_part:
                            content_delta_str = content_part["text"] or ""
                        elif isinstance(content_part, str):
                            content_delta_str = content_part

                    # Handle tool call deltas
                    if (
                        hasattr(llama_chunk_completion_message, "tool_calls")
                        and llama_chunk_completion_message.tool_calls
                    ):
                        for tc_delta in llama_chunk_completion_message.tool_calls:
                            tool_id = getattr(tc_delta, "id", None)
                            if not tool_id:  # Should always have an ID from Llama API
                                logger.warning("Tool call delta missing ID, skipping.")
                                continue

                            # Initialize buffer for this tool_id if first time seen
                            if tool_id not in aggregated_tool_calls_buffer:
                                aggregated_tool_calls_buffer[tool_id] = {
                                    "id": tool_id,
                                    "name": None,
                                    "args_str": "",  # Accumulated arguments string
                                    "index": next_tool_call_chunk_index,  # LangChain's index for the tool call
                                }
                                next_tool_call_chunk_index += 1

                            buffer_entry = aggregated_tool_calls_buffer[tool_id]

                            # Update name if present in delta
                            delta_func = getattr(tc_delta, "function", None)
                            if (
                                delta_func
                                and hasattr(delta_func, "name")
                                and delta_func.name
                            ):
                                buffer_entry["name"] = delta_func.name

                            # Accumulate arguments string
                            args_delta_str = ""
                            if (
                                delta_func
                                and hasattr(delta_func, "arguments")
                                and delta_func.arguments
                            ):
                                args_delta_str = delta_func.arguments
                                buffer_entry["args_str"] += args_delta_str

                            # Create a LangChain ToolCallChunk for this specific delta
                            # The 'args' in ToolCallChunk is the partial string for this chunk
                            lc_tool_chunk = ToolCallChunk(
                                name=buffer_entry[
                                    "name"
                                ],  # Use current name from buffer
                                args=args_delta_str,  # Pass the arguments delta from this chunk
                                id=tool_id,
                                index=buffer_entry["index"],
                            )
                            tool_call_chunks_for_lc.append(lc_tool_chunk)
                            generation_info["finish_reason"] = (
                                "tool_calls"  # Mark that a tool call is in progress/occurred
                            )
                            content_delta_str = ""  # If there are tool calls, content should be empty for this AIMessageChunk part

                # Fallback: Textual tool call parsing from content_delta_str if no structured tool_calls
                # This should ideally happen *after* structured tool calls are processed for a chunk.
                # If structured tool calls were found, content_delta_str would be empty.
                if (
                    not tool_call_chunks_for_lc
                    and prepared_llm_tools
                    and content_delta_str
                ):
                    # Log the content we're trying to match against for debugging
                    logger.debug(
                        f"Looking for textual tool call in content: '{content_delta_str}'"
                    )

                    # Use a simpler regex pattern that's more tolerant of spacing
                    match = re.search(
                        r"\[\s*([a-zA-Z0-9_]+)\s*(?:\(\s*(.*?)\s*\))?\s*\]",
                        content_delta_str,
                    )
                    if match:
                        tool_name_from_text = match.group(1)
                        args_str_from_text = (
                            match.group(2) if match.group(2) is not None else ""
                        )

                        logger.debug(
                            f"Found textual tool call match: name='{tool_name_from_text}', args='{args_str_from_text}'"
                        )

                        # Get the list of available tools, with better defensive checks
                        available_tool_names = []
                        if prepared_llm_tools:
                            for t in prepared_llm_tools:
                                if isinstance(t, dict) and "function" in t:
                                    func_dict = t.get("function", {})
                                    if (
                                        isinstance(func_dict, dict)
                                        and "name" in func_dict
                                    ):
                                        available_tool_names.append(func_dict["name"])

                        logger.debug(f"Available tool names: {available_tool_names}")

                        # If the matched tool name is one of our available tools, create a ToolCallChunk
                        if tool_name_from_text in available_tool_names:
                            logger.info(
                                f"Detected textual tool call in stream: {tool_name_from_text}"
                            )
                            textual_tool_id = str(uuid.uuid4())

                            # Since this is a textual match, it's assumed to be a complete call in this chunk's content.
                            # We add it to the buffer as if it's a new tool call.
                            if (
                                textual_tool_id not in aggregated_tool_calls_buffer
                            ):  # Should be unique
                                aggregated_tool_calls_buffer[textual_tool_id] = {
                                    "id": textual_tool_id,
                                    "name": tool_name_from_text,
                                    "args_str": args_str_from_text,  # Full args from textual parse
                                    "index": next_tool_call_chunk_index,
                                }
                                next_tool_call_chunk_index += 1

                            # For LangChain, we need to create a dictionary that looks like a ToolCallChunk
                            # but with the right structure for AIMessageChunk.tool_call_chunks
                            lc_textual_tool_dict = {
                                "name": tool_name_from_text,
                                "args": args_str_from_text,  # The full args string for this textual tool call
                                "id": textual_tool_id,
                                "index": aggregated_tool_calls_buffer[textual_tool_id][
                                    "index"
                                ],
                            }
                            tool_call_chunks_for_lc.append(lc_textual_tool_dict)
                            content_delta_str = (
                                ""  # Content is consumed by the textual tool call
                            )
                            generation_info["finish_reason"] = "tool_calls"
                        else:
                            logger.warning(
                                f"Textual tool call '{tool_name_from_text}' found in content, but not in available tools: {available_tool_names}"
                            )

                # --- Metadata from the main chunk object ---
                if hasattr(chunk, "model") and chunk.model:
                    generation_info["model_name"] = chunk.model
                if (
                    hasattr(chunk, "stop_reason") and chunk.stop_reason
                ):  # Llama API specific
                    generation_info["finish_reason"] = chunk.stop_reason
                if chunk_dict.get("x_request_id"):  # From to_dict()
                    generation_info["x_request_id"] = chunk_dict.get("x_request_id")

                # --- Usage Metadata from Llama API chunk ---
                current_chunk_usage_metadata = None
                if hasattr(chunk, "usage") and chunk.usage:
                    usage_data = chunk.usage
                    prompt_tokens_val = getattr(usage_data, "prompt_tokens", 0)
                    completion_tokens_val = getattr(usage_data, "completion_tokens", 0)
                    total_tokens_val = getattr(usage_data, "total_tokens", 0)

                    current_chunk_usage_metadata = UsageMetadata(
                        input_tokens=prompt_tokens_val,
                        output_tokens=completion_tokens_val,  # This is per-chunk, sum later
                        total_tokens=total_tokens_val,  # This is likely cumulative from API
                    )
                    generation_info[
                        "usage_metadata"
                    ] = {  # For ChatGenerationChunk.generation_info
                        "input_tokens": prompt_tokens_val,
                        "output_tokens": completion_tokens_val,
                        "total_tokens": total_tokens_val,
                    }

                # Create AIMessageChunk for LangChain
                lc_message_chunk = AIMessageChunk(
                    content=content_delta_str,
                    tool_call_chunks=tool_call_chunks_for_lc
                    if tool_call_chunks_for_lc
                    else [],  # Ensure it's a list
                    usage_metadata=current_chunk_usage_metadata,  # Attach to AIMessageChunk
                )

                # Create ChatGenerationChunk for LangChain
                lc_chat_generation_chunk = ChatGenerationChunk(
                    message=lc_message_chunk,
                    generation_info=generation_info
                    if generation_info
                    else None,  # Pass along generation_info
                )

                all_chunks_for_callback.append(lc_chat_generation_chunk)

                if run_manager:
                    # Ensure we only pass the string content to on_llm_new_token
                    token_content = (
                        lc_message_chunk.content
                        if isinstance(lc_message_chunk.content, str)
                        else ""
                    )
                    run_manager.on_llm_new_token(
                        token=token_content,  # Pass text delta
                        chunk=lc_chat_generation_chunk,  # Pass the full LangChain chunk
                    )
                yield lc_chat_generation_chunk

            # After the loop, process aggregated tool calls for the final callback
            if run_manager:
                final_lc_content = "".join(
                    chunk.message.content
                    for chunk in all_chunks_for_callback
                    if isinstance(chunk.message.content, str)
                )

                final_lc_tool_calls: List[Dict[str, Any]] = []
                for tool_id, data in aggregated_tool_calls_buffer.items():
                    parsed_args = {}
                    try:
                        if data["args_str"]:
                            parsed_args = json.loads(data["args_str"])
                        # Ensure args is always a dict for LangChain
                        if not isinstance(parsed_args, dict):
                            parsed_args = {"value": parsed_args}  # Wrap if not dict
                    except json.JSONDecodeError:
                        logger.warning(
                            f"Failed to parse JSON for tool args: {data['args_str']}. Using raw string or fallback parser."
                        )
                        # Attempt fallback parsing for malformed args like name="value"
                        parsed_args = _parse_textual_tool_args(
                            data["args_str"]
                        )  # _parse_textual_tool_args returns a dict
                        if not isinstance(parsed_args, dict):  # Ensure it's a dict
                            parsed_args = {"value": data["args_str"]}

                    final_lc_tool_calls.append(
                        {
                            "id": data["id"],
                            "name": data["name"] or "unknown_tool",
                            "args": parsed_args,
                            "type": "function",  # LangChain expects this
                        }
                    )

                # If final_lc_tool_calls were generated, the content might be empty
                # or it might be a textual representation that was superseded.
                # Clear final_lc_content if it solely represents a tool call that is now structured.
                if final_lc_tool_calls and final_lc_content:
                    if re.fullmatch(
                        r"\s*\[\s*([a-zA-Z0-9_]+)\s*(?:\(\s*(.*?)\s*\))?\s*\]\s*",
                        final_lc_content,
                    ):
                        final_lc_content = ""

                final_lc_message = AIMessage(
                    content=final_lc_content, tool_calls=final_lc_tool_calls
                )

                # Aggregate Usage Metadata for the final AIMessage
                total_input_tokens_cb = 0
                total_output_tokens_cb = 0
                # Get input tokens from the first chunk that has it (should be consistent)
                for chunk_cb in all_chunks_for_callback:
                    # Safely access usage_metadata
                    usage_meta = getattr(chunk_cb.message, "usage_metadata", None)
                    if usage_meta and getattr(usage_meta, "input_tokens", 0) > 0:
                        total_input_tokens_cb = getattr(usage_meta, "input_tokens")
                        break
                # Sum output tokens from all chunks
                for chunk_cb in all_chunks_for_callback:
                    usage_meta = getattr(chunk_cb.message, "usage_metadata", None)
                    if usage_meta and getattr(usage_meta, "output_tokens", 0) > 0:
                        total_output_tokens_cb += getattr(usage_meta, "output_tokens")

                if total_input_tokens_cb > 0 or total_output_tokens_cb > 0:
                    final_lc_message.usage_metadata = UsageMetadata(
                        input_tokens=total_input_tokens_cb,
                        output_tokens=total_output_tokens_cb,
                        total_tokens=total_input_tokens_cb + total_output_tokens_cb,
                    )

                final_generation_info_cb = (
                    all_chunks_for_callback[-1].generation_info
                    if all_chunks_for_callback
                    and all_chunks_for_callback[-1].generation_info
                    else {}
                )
                if (
                    final_lc_tool_calls
                    and final_generation_info_cb.get("finish_reason") != "tool_calls"
                ):
                    final_generation_info_cb["finish_reason"] = "tool_calls"

                llm_result_for_callback = LLMResult(
                    generations=[
                        [
                            ChatGeneration(
                                message=final_lc_message,
                                generation_info=final_generation_info_cb,
                            )
                        ]
                    ],
                    llm_output={  # Standardize this
                        "model_name": self.model_name,
                        "token_usage": {
                            "prompt_tokens": total_input_tokens_cb,
                            "completion_tokens": total_output_tokens_cb,
                            "total_tokens": total_input_tokens_cb
                            + total_output_tokens_cb,
                        },
                        "finish_reason": final_generation_info_cb.get("finish_reason"),
                        "x_request_id": final_generation_info_cb.get("x_request_id"),
                    },
                )
                run_manager.on_llm_end(llm_result_for_callback)

        except Exception as e:
            logger.error(f"Error in _stream: {e}", exc_info=True)
            if run_manager:
                run_manager.on_llm_error(e)
            raise
