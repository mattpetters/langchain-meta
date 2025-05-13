import json
import logging
import uuid
import re
from datetime import datetime
from typing import (
    Any,
    AsyncIterator,
    Callable,
    Dict,
    List,
    Literal,
    Optional,
    Type,
    Union,
    cast,
    Tuple,
)

from langchain_core.callbacks.manager import AsyncCallbackManagerForLLMRun
from langchain_core.messages import (
    AIMessage,
    AIMessageChunk,
    BaseMessage,  # Used by _detect_supervisor_request if that were moved, but it's on self
    ToolCallChunk,
)
from langchain_core.messages.ai import UsageMetadata
from langchain_core.outputs import (
    ChatGeneration,
    ChatGenerationChunk,
    ChatResult,
    LLMResult,
    Generation,
)
from langchain_core.tools import BaseTool
from llama_api_client import AsyncLlamaAPIClient
from llama_api_client.types.chat import (
    completion_create_params,
)
from llama_api_client.types.create_chat_completion_response import (
    CreateChatCompletionResponse,
)
from pydantic import BaseModel

# Assuming chat_models.py is in langchain_meta.chat_models
# Adjust the import path if necessary based on your project structure.
from .serialization import (
    _lc_tool_to_llama_tool_param,
    _parse_textual_tool_args,
)  # Changed from ..chat_models
from ..utils import parse_malformed_args_string  # Import from main utils

logger = logging.getLogger(__name__)


class AsyncChatMetaLlamaMixin:
    """Mixin class to hold asynchronous methods for ChatMetaLlama."""

    # Type hints for attributes/methods from ChatMetaLlama main class
    # that are used by these async methods via \`self\`.
    _async_client: Optional[AsyncLlamaAPIClient]
    model_name: str

    # These methods are expected to be part of the main ChatMetaLlama class
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

    def _detect_supervisor_request(self, messages: List[BaseMessage]) -> bool:
        raise NotImplementedError  # pragma: no cover

    async def _agenerate(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[AsyncCallbackManagerForLLMRun] = None,
        tools: Optional[List[Union[Dict, Type[BaseModel], Callable, BaseTool]]] = None,
        tool_choice: Optional[
            Union[dict, str, Literal["auto", "none", "any", "required"]]
        ] = None,
        **kwargs: Any,
    ) -> ChatResult:
        """Asynchronously generates a chat response using AsyncLlamaAPIClient."""
        self._ensure_client_initialized()
        if not self._async_client:
            raise ValueError(
                "Async client not initialized. Call \\`async_init_clients\\` first."
            )
        async_client_to_use = self._async_client

        prompt_tokens = 0
        completion_tokens = 0
        start_time = datetime.now()

        if tool_choice is not None and "tool_choice" not in kwargs:
            kwargs["tool_choice"] = tool_choice

        if kwargs.get("stream", False):
            return await self._astream_with_aggregation_and_retries(
                messages=messages,
                stop=stop,
                run_manager=run_manager,
                tools=tools,
                tool_choice=tool_choice,
                **kwargs,
            )

        logger.debug(f"_agenerate received direct tools: {tools}")
        logger.debug(f"_agenerate received direct tool_choice: {tool_choice}")
        logger.debug(f"_agenerate received kwargs: {kwargs}")

        effective_tools_lc_input = tools
        if effective_tools_lc_input is None and "tools" in kwargs:
            effective_tools_lc_input = kwargs.get("tools")
            logger.debug(
                "_agenerate (non-streaming): Using 'tools' from **kwargs as direct 'tools' parameter was None."
            )

        prepared_llm_tools: Optional[List[completion_create_params.Tool]] = None
        if effective_tools_lc_input:
            tools_list_to_prepare = effective_tools_lc_input
            if not isinstance(effective_tools_lc_input, list):
                logger.warning(
                    f"_agenerate (non-streaming): effective_tools_lc_input was not a list ({type(effective_tools_lc_input)}). Wrapping in a list."
                )
                tools_list_to_prepare = [effective_tools_lc_input]

            prepared_llm_tools = [
                _lc_tool_to_llama_tool_param(tool) for tool in tools_list_to_prepare
            ]
        else:
            logger.debug("_agenerate (non-streaming): No effective tools to prepare.")

        final_kwargs_for_prepare = kwargs.copy()
        final_kwargs_for_prepare.pop("tools", None)

        if tool_choice is not None:
            final_kwargs_for_prepare["tool_choice"] = tool_choice

        api_params = self._prepare_api_params(
            messages, tools=prepared_llm_tools, **final_kwargs_for_prepare
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
                    f"_agenerate: Prepared structured output metadata: {structured_output_metadata}"
                )

        # Callback triggering (on_chat_model_start) should be handled by the base class before _agenerate
        # We just need to ensure the parameters are correctly gathered if needed elsewhere,
        # but we don't trigger the callback here.
        # invocation_params = self._get_invocation_params(api_params=api_params, **final_kwargs_for_prepare)

        logger.debug(f"Llama API (async) Request (ainvoke): {api_params}")
        try:
            call_result: CreateChatCompletionResponse = (
                await async_client_to_use.chat.completions.create(**api_params)
            )
            logger.debug(f"Llama API (async) Response (ainvoke): {call_result}")
        except Exception as e:
            if run_manager:
                try:
                    if hasattr(run_manager, "on_llm_error"):
                        await run_manager.on_llm_error(error=e)
                except Exception as callback_err:
                    logger.warning(f"Error in LangSmith error callback: {callback_err}")
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

        generation_info = {}
        tool_calls_data = []
        if result_msg and hasattr(result_msg, "tool_calls") and result_msg.tool_calls:
            processed_tool_calls: List[Dict] = []
            for idx, tc in enumerate(result_msg.tool_calls):
                tc_id = (
                    getattr(tc, "id", None) or f"llama_tc_{idx}" or str(uuid.uuid4())
                )
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
                    final_args = {"value": tc_args_str}
                except Exception as e:
                    logger.warning(
                        f"Unexpected error processing tool call arguments: {e}. Representing as string."
                    )
                    final_args = {"value": tc_args_str}
                # Defensive: always ensure id, name, args
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

        # Fallback: If no tool_calls from API, try to parse from content_str if tools were provided and content looks like a textual tool call
        if not tool_calls_data and content_str and prepared_llm_tools:
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
                            parsed_args = _parse_textual_tool_args(
                                args_str_from_content
                            )
                        except Exception as e:
                            logger.warning(
                                f"Failed to parse arguments '{args_str_from_content}' for textual tool call '{tool_name_from_content}': {e}. Using raw string as arg."
                            )
                            parsed_args = {"value": args_str_from_content}
                    # Defensive: always ensure id, name, args
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
                            "type": "function",
                        }
                    )
                    content_str = (
                        ""  # Clear content as it was a tool call representation
                    )
                    generation_info["finish_reason"] = "tool_calls"
                else:
                    logger.warning(
                        f"Textual tool call '{tool_name_from_content}' found in content, but not in available tools: {available_tool_names}"
                    )
            else:
                logger.debug(
                    f"Content '{content_str}' did not match textual tool call pattern."
                )

        message = AIMessage(content=content_str or "", tool_calls=tool_calls_data)

        if result_msg and hasattr(result_msg, "stop_reason") and result_msg.stop_reason:
            generation_info["finish_reason"] = result_msg.stop_reason
        elif hasattr(call_result, "stop_reason") and getattr(
            call_result, "stop_reason", None
        ):
            generation_info["finish_reason"] = getattr(call_result, "stop_reason")

        if (
            hasattr(call_result, "metrics")
            and call_result.metrics
            and isinstance(call_result.metrics, list)
        ):
            usage_meta = {}
            for metric_item in call_result.metrics:
                if hasattr(metric_item, "metric") and hasattr(metric_item, "value"):
                    metric_name = getattr(metric_item, "metric")
                    metric_value = (
                        int(metric_item.value) if metric_item.value is not None else 0
                    )
                    if metric_name == "num_prompt_tokens":
                        usage_meta["input_tokens"] = metric_value
                        prompt_tokens = metric_value
                    elif metric_name == "num_completion_tokens":
                        usage_meta["output_tokens"] = metric_value
                        completion_tokens = metric_value
                    elif metric_name == "num_total_tokens":
                        usage_meta["total_tokens"] = metric_value
            if usage_meta:
                generation_info["usage_metadata"] = usage_meta
                if hasattr(message, "usage_metadata"):
                    try:
                        constructed_usage = UsageMetadata(
                            input_tokens=usage_meta.get("input_tokens", 0),
                            output_tokens=usage_meta.get("output_tokens", 0),
                            total_tokens=usage_meta.get("total_tokens", 0),
                        )
                        message.usage_metadata = constructed_usage
                    except Exception as e:
                        logger.warning(f"Could not construct UsageMetadata: {e}")
        elif hasattr(call_result, "usage") and getattr(call_result, "usage", None):
            usage_data = getattr(call_result, "usage")
            input_tokens_val = int(getattr(usage_data, "prompt_tokens", 0))
            output_tokens_val = int(getattr(usage_data, "completion_tokens", 0))
            total_tokens_val = int(getattr(usage_data, "total_tokens", 0))
            usage_meta = {
                "input_tokens": input_tokens_val,
                "output_tokens": output_tokens_val,
                "total_tokens": total_tokens_val,
            }
            prompt_tokens = usage_meta["input_tokens"]
            completion_tokens = usage_meta["output_tokens"]
            total_tokens = usage_meta["total_tokens"]
            if any(usage_meta.values()):
                generation_info["usage_metadata"] = usage_meta
                if hasattr(message, "usage_metadata"):
                    try:
                        constructed_usage = UsageMetadata(
                            input_tokens=usage_meta.get("input_tokens", 0),
                            output_tokens=usage_meta.get("output_tokens", 0),
                            total_tokens=usage_meta.get("total_tokens", 0),
                        )
                        message.usage_metadata = constructed_usage
                    except Exception as e:
                        logger.warning(f"Could not construct UsageMetadata: {e}")

        if hasattr(call_result, "x_request_id") and getattr(
            call_result, "x_request_id", None
        ):
            generation_info["x_request_id"] = getattr(call_result, "x_request_id")
        generation_info["response_metadata"] = call_result.to_dict()
        generation_info["llm_output"] = call_result.to_dict()

        end_time = datetime.now()
        duration = (end_time - start_time).total_seconds()
        generation_info["duration"] = duration

        # Standardize llm_output structure for callbacks
        llm_output_data = {
            "model_name": self.model_name,
            "token_usage": {  # Ensure this structure exists
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "total_tokens": prompt_tokens + completion_tokens,
            },
            "system_fingerprint": getattr(call_result, "system_fingerprint", None),
            "request_id": getattr(call_result, "x_request_id", None),
            "finish_reason": generation_info.get("finish_reason"),
            # Keep the raw response for detailed inspection if needed
            "raw_response": call_result.to_dict(),
        }
        # Ensure generation_info also has consistent token usage if needed elsewhere
        # We'll use the same dict structure as llm_output for consistency
        generation_info["token_usage"] = llm_output_data["token_usage"]

        result = ChatResult(
            generations=[
                ChatGeneration(message=message, generation_info=generation_info)
            ]
        )
        result.llm_output = llm_output_data  # Use the standardized structure

        if run_manager:
            if hasattr(run_manager, "on_llm_end"):
                # Construct LLMResult for the callback
                generations_for_llm_result: List[
                    List[
                        Union[
                            Generation,
                            ChatGeneration,
                            ChatGenerationChunk,
                            ChatGenerationChunk,
                        ]
                    ]
                ] = [
                    cast(
                        List[
                            Union[
                                Generation,
                                ChatGeneration,
                                ChatGenerationChunk,
                                ChatGenerationChunk,
                            ]
                        ],
                        result.generations,
                    )
                ]
                llm_result_for_callback = LLMResult(
                    generations=generations_for_llm_result,
                    llm_output=result.llm_output,
                    run=None,
                )
                await run_manager.on_llm_end(llm_result_for_callback)
        return result

    async def _astream(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[AsyncCallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> AsyncIterator[ChatGenerationChunk]:
        """Asynchronously streams chat responses using AsyncLlamaAPIClient."""
        self._ensure_client_initialized()
        if self._async_client is None:
            raise ValueError("AsyncLlamaAPIClient not initialized.")

        active_client = kwargs.get("async_client") or self._async_client
        if not active_client:
            raise ValueError("Could not obtain an active AsyncLlamaAPIClient.")

        effective_tools_lc_input = kwargs.get("tools")
        if (
            effective_tools_lc_input is None and "tools" in kwargs
        ):  # Check if tools came from .bind()
            effective_tools_lc_input = kwargs.get("tools")
            logger.debug(
                "_astream: Using 'tools' from **kwargs as direct 'tools' parameter was None."
            )

        prepared_llm_tools: Optional[List[completion_create_params.Tool]] = None
        if effective_tools_lc_input:
            tools_list_to_prepare = effective_tools_lc_input
            if not isinstance(effective_tools_lc_input, list):
                logger.warning(
                    f"_astream: effective_tools_lc_input was not a list ({type(effective_tools_lc_input)}). Wrapping in a list."
                )
                tools_list_to_prepare = [effective_tools_lc_input]
            prepared_llm_tools = [
                _lc_tool_to_llama_tool_param(tool) for tool in tools_list_to_prepare
            ]
        else:
            logger.debug("_astream: No effective tools to prepare.")

        final_kwargs_for_prepare = kwargs.copy()
        final_kwargs_for_prepare.pop("tools", None)

        api_params = self._prepare_api_params(
            messages,
            tools=prepared_llm_tools,
            stop=stop,
            stream=True,
            **final_kwargs_for_prepare,
        )
        # If streaming, llama-api-client might not support tool_choice in create()
        if api_params.get("stream"):
            api_params.pop(
                "tool_choice", None
            )  # Remove tool_choice if present for streaming

        logger.debug(f"Llama API (async stream) Request: {api_params}")

        all_chunks_for_callback: List[ChatGenerationChunk] = []
        aggregated_tool_calls_buffer: Dict[
            str, Dict[str, Any]
        ] = {}  # tool_id -> {'id', 'name', 'args_str', 'index'}
        next_tool_call_chunk_index = 0  # For LangChain's ToolCallChunk.index

        try:
            async for chunk in await active_client.chat.completions.create(
                **api_params
            ):
                # logger.debug(f"Llama API (async stream) Stream Chunk: {chunk.to_dict()}") # Reduce noise

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
                                logger.warning(
                                    "(Async) Tool call delta missing ID, skipping."
                                )
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
                if (
                    not tool_call_chunks_for_lc
                    and prepared_llm_tools
                    and content_delta_str
                ):
                    # Log the content we're trying to match against for debugging
                    logger.debug(
                        f"(Async) Looking for textual tool call in content: '{content_delta_str}'"
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
                            f"(Async) Found textual tool call match: name='{tool_name_from_text}', args='{args_str_from_text}'"
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

                        logger.debug(
                            f"(Async) Available tool names: {available_tool_names}"
                        )

                        # If the matched tool name is one of our available tools, create a ToolCallChunk
                        if tool_name_from_text in available_tool_names:
                            logger.info(
                                f"(Async) Detected textual tool call in stream: {tool_name_from_text}"
                            )
                            textual_tool_id = str(uuid.uuid4())

                            # Add to buffer for consistency, assuming complete call in this chunk
                            if textual_tool_id not in aggregated_tool_calls_buffer:
                                aggregated_tool_calls_buffer[textual_tool_id] = {
                                    "id": textual_tool_id,
                                    "name": tool_name_from_text,
                                    "args_str": args_str_from_text,
                                    "index": next_tool_call_chunk_index,
                                }
                                next_tool_call_chunk_index += 1

                            # For LangChain, we need to create a dictionary that looks like a ToolCallChunk
                            # but with the right structure for AIMessageChunk.tool_call_chunks
                            lc_textual_tool_dict = {
                                "name": tool_name_from_text,
                                "args": args_str_from_text,
                                "id": textual_tool_id,
                                "index": aggregated_tool_calls_buffer[textual_tool_id][
                                    "index"
                                ],
                            }
                            tool_call_chunks_for_lc.append(lc_textual_tool_dict)
                            content_delta_str = ""  # Content consumed
                            generation_info["finish_reason"] = "tool_calls"
                        else:
                            logger.warning(
                                f"(Async) Textual tool call '{tool_name_from_text}' found in content, but not in available tools: {available_tool_names}"
                            )

                # --- Metadata from the main chunk object ---
                if hasattr(chunk, "model") and chunk.model:
                    generation_info["model_name"] = chunk.model
                if hasattr(chunk, "stop_reason") and chunk.stop_reason:
                    generation_info["finish_reason"] = chunk.stop_reason
                if chunk_dict.get("x_request_id"):
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
                        output_tokens=completion_tokens_val,
                        total_tokens=total_tokens_val,
                    )
                    generation_info["usage_metadata"] = {
                        "input_tokens": prompt_tokens_val,
                        "output_tokens": completion_tokens_val,
                        "total_tokens": total_tokens_val,
                    }

                # Create AIMessageChunk for LangChain
                lc_message_chunk = AIMessageChunk(
                    content=content_delta_str,
                    tool_call_chunks=tool_call_chunks_for_lc
                    if tool_call_chunks_for_lc
                    else [],
                    usage_metadata=current_chunk_usage_metadata,
                )

                # Create ChatGenerationChunk for LangChain
                lc_chat_generation_chunk = ChatGenerationChunk(
                    message=lc_message_chunk,
                    generation_info=generation_info if generation_info else None,
                )

                all_chunks_for_callback.append(lc_chat_generation_chunk)

                yield lc_chat_generation_chunk

                if run_manager:
                    await run_manager.on_llm_new_token(
                        token=lc_message_chunk.content
                        if isinstance(lc_message_chunk.content, str)
                        else "",
                        chunk=lc_chat_generation_chunk,
                    )

            # Process aggregated results for the final on_llm_end callback
            if run_manager:
                final_lc_content_cb = "".join(
                    chunk.message.content
                    for chunk in all_chunks_for_callback
                    if isinstance(chunk.message.content, str)
                )
                final_lc_tool_calls_cb: List[Dict[str, Any]] = []
                for tool_id, data in aggregated_tool_calls_buffer.items():
                    parsed_args_cb = {}
                    try:
                        if data["args_str"]:
                            parsed_args_cb = json.loads(data["args_str"])
                        if not isinstance(parsed_args_cb, dict):
                            parsed_args_cb = {"value": parsed_args_cb}
                    except json.JSONDecodeError:
                        logger.warning(
                            f"(Async) Failed JSON parse for tool args: {data['args_str']}. Using fallback."
                        )
                        parsed_args_cb = _parse_textual_tool_args(data["args_str"])
                        if not isinstance(parsed_args_cb, dict):
                            parsed_args_cb = {"value": data["args_str"]}

                    final_lc_tool_calls_cb.append(
                        {
                            "id": data["id"],
                            "name": data["name"] or "unknown_tool",
                            "args": parsed_args_cb,
                            "type": "function",
                        }
                    )

                # Clear content if it was purely a textual tool call representation
                if final_lc_tool_calls_cb and final_lc_content_cb:
                    if re.fullmatch(
                        r"\s*\[\s*([a-zA-Z0-9_]+)\s*(?:\(\s*(.*?)\s*\))?\s*\]\s*",
                        final_lc_content_cb,
                    ):
                        final_lc_content_cb = ""

                final_lc_message_cb = AIMessage(
                    content=final_lc_content_cb,
                    tool_calls=final_lc_tool_calls_cb
                    if final_lc_tool_calls_cb
                    else None,
                )

                # Aggregate Usage for callback
                total_input_tokens_cb = 0
                total_output_tokens_cb = 0
                for chunk_cb in all_chunks_for_callback:
                    usage_meta_cb = getattr(chunk_cb.message, "usage_metadata", None)
                    if usage_meta_cb:
                        total_input_tokens_cb = max(
                            total_input_tokens_cb,
                            getattr(usage_meta_cb, "input_tokens", 0),
                        )
                        total_output_tokens_cb += getattr(
                            usage_meta_cb, "output_tokens", 0
                        )

                if total_input_tokens_cb > 0 or total_output_tokens_cb > 0:
                    final_lc_message_cb.usage_metadata = UsageMetadata(
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
                    final_lc_tool_calls_cb
                    and final_generation_info_cb.get("finish_reason") != "tool_calls"
                ):
                    final_generation_info_cb["finish_reason"] = "tool_calls"

                # Create LLMResult for on_llm_end
                llm_result_for_callback = LLMResult(
                    generations=[
                        [
                            ChatGeneration(
                                message=final_lc_message_cb,
                                generation_info=final_generation_info_cb,
                            )
                        ]
                    ],
                    llm_output={
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
                await run_manager.on_llm_end(llm_result_for_callback)

        except Exception as e:
            logger.error(f"(Async) Error in _astream loop: {e}", exc_info=True)
            if run_manager:
                await run_manager.on_llm_error(e)
            raise

    async def _astream_with_aggregation_and_retries(
        self,
        messages: List[BaseMessage],
        run_manager: Optional[AsyncCallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> ChatResult:
        final_content = ""
        tool_deltas_by_index: Dict[int, Dict[str, Any]] = {}
        final_generation_info = {}
        chunk_index_to_tool_calls = {}
        final_tool_calls = []
        prompt_tokens = 0
        completion_tokens = 0
        total_tokens = 0
        final_generation_info_aggregated = {}

        async for chunk in self._astream(
            messages=messages, run_manager=run_manager, **kwargs
        ):
            final_content_for_current_chunk_message = (
                chunk.text
            )  # text already aggregates content for this specific chunk

            if chunk.generation_info:
                final_generation_info.update(chunk.generation_info)

            # Process tool_call_chunks from the current yielded chunk
            # These are already in LangChain's ToolCallChunk format
            # AIMessageChunk wants a list of these chunk dicts
            tool_call_chunks_for_aimessagechunk = []
            if chunk.message and chunk.message.tool_call_chunks:
                # Explicitly cast for the loop context if isinstance check isn't enough
                message_chunk = cast(AIMessageChunk, chunk.message)
                for tc_chunk in message_chunk.tool_call_chunks:
                    tc_index = getattr(tc_chunk, "index", None)
                    if tc_index is None:
                        logger.warning(
                            "(Async) Tool call delta missing index, skipping."
                        )
                        continue

                    if tc_index not in tool_deltas_by_index:
                        tool_deltas_by_index[tc_index] = {
                            "id": tc_chunk.id,
                            "name": tc_chunk.name,
                            "args_chunks": [],
                            "type": "function",
                        }

                    # Append args to the accumulated args for this index
                    args_val = tc_chunk.args
                    if args_val:
                        tool_deltas_by_index[tc_index]["args_chunks"].append(args_val)

            # Accumulate content chunks
            if hasattr(chunk.message, "content") and chunk.message.content:
                # Ensure content is a string before adding
                content_val = chunk.message.content
                if isinstance(content_val, str):
                    final_content += content_val
                elif isinstance(content_val, dict) and "text" in content_val:
                    if isinstance(content_val["text"], str):
                        final_content += content_val["text"]
                # else: log warning or handle other types if necessary

            # Accumulate usage metadata (taking the last complete one)
            if (
                getattr(chunk, "generation_info", None)  # Use getattr for safe access
            ):
                gen_info = (
                    chunk.generation_info
                )  # Access after check (or keep using getattr)
                if "finish_reason" in gen_info:
                    final_generation_info_aggregated["finish_reason"] = gen_info[
                        "finish_reason"
                    ]
                if "usage_metadata" in gen_info:
                    # Get token counts from usage
                    usage = gen_info["usage_metadata"]
                    if "input_tokens" in usage and usage["input_tokens"]:
                        prompt_tokens = max(prompt_tokens, int(usage["input_tokens"]))
                    if "output_tokens" in usage and usage["output_tokens"]:
                        completion_tokens += int(usage["output_tokens"])
                    if "total_tokens" in usage and usage["total_tokens"]:
                        total_tokens = max(total_tokens, int(usage["total_tokens"]))

        # Process the accumulated tool call chunks to build complete tool calls
        for idx, tc_data in tool_deltas_by_index.items():
            args_str = "".join(tc_data["args_chunks"])
            # Parse args if needed
            try:
                args_dict = json.loads(args_str)
            except json.JSONDecodeError:
                # Fall back to the string parsing utility
                args_dict = _parse_textual_tool_args(args_str)

            final_tool_calls.append(
                {
                    "id": tc_data["id"],
                    "name": tc_data["name"],
                    "args": args_dict,
                    "type": "function",
                }
            )

        # If there were tool calls, set finish_reason
        if final_tool_calls:
            final_generation_info_aggregated["finish_reason"] = "tool_calls"
            # Clear content if it appears to be a tool call in text form
            if final_content and re.fullmatch(
                r"\s*\[\s*([a-zA-Z0-9_]+)\s*(?:\(\s*(.*?)\s*\))?\s*\]\s*", final_content
            ):
                final_content = ""

        # If usage info wasn't in generation_info, calculate something reasonable
        if not total_tokens:
            total_tokens = prompt_tokens + completion_tokens

        # Special case: If completion_tokens is 0 but we have content, estimate
        if completion_tokens == 0 and final_content:
            completion_tokens = len(final_content) // 4  # Rough estimate
            total_tokens = prompt_tokens + completion_tokens

        # Create final AIMessage
        final_message = AIMessage(
            content=final_content,
            tool_calls=final_tool_calls if final_tool_calls else None,
        )

        # Add usage metadata to the message
        try:
            if prompt_tokens or completion_tokens:
                final_message.usage_metadata = UsageMetadata(
                    input_tokens=prompt_tokens,
                    output_tokens=completion_tokens,
                    total_tokens=total_tokens,
                )
        except Exception as e:
            logger.warning(f"Could not construct UsageMetadata for final message: {e}")

        # Debug logging
        logger.debug(
            f"_aget_stream_results: final_message.content: '{final_message.content}'"
        )
        logger.debug(
            f"_aget_stream_results: final_message.tool_calls: {final_message.tool_calls}"
        )

        # Return a ChatResult with the final message
        return ChatResult(
            generations=[
                ChatGeneration(
                    message=final_message,
                    generation_info=final_generation_info_aggregated,
                )
            ],
            llm_output={
                "token_usage": {
                    "prompt_tokens": prompt_tokens,
                    "completion_tokens": completion_tokens,
                    "total_tokens": total_tokens,
                },
                "model_name": self.model_name,
            },
        )

    async def _aget_stream_results(
        self,
        chunks: List[ChatGenerationChunk],
        run_manager: Optional[AsyncCallbackManagerForLLMRun] = None,
    ) -> ChatResult:
        """Process the streaming chunks into a consolidated ChatResult."""
        # If there are no chunks, return an empty result
        if not chunks:
            return ChatResult(generations=[])

        # Get the final message by merging all chunks
        final_content = "".join(
            chunk.message.content if isinstance(chunk.message.content, str) else ""
            for chunk in chunks
        )

        # Merge all token usage information
        prompt_tokens = 0
        completion_tokens = 0
        final_generation_info = {}

        for chunk in chunks:
            # Get token info from generation_info if available
            if chunk.generation_info:
                if "usage_metadata" in chunk.generation_info:
                    usage = chunk.generation_info["usage_metadata"]
                    if "input_tokens" in usage and usage["input_tokens"]:
                        prompt_tokens = max(prompt_tokens, int(usage["input_tokens"]))
                    if "output_tokens" in usage and usage["output_tokens"]:
                        completion_tokens += int(usage["output_tokens"])
                # Get other generation info from the last chunk
                if "finish_reason" in chunk.generation_info:
                    final_generation_info["finish_reason"] = chunk.generation_info[
                        "finish_reason"
                    ]

        # Get all tool calls by merging tool call chunks
        # Group by tool_id
        tool_calls_by_id = {}
        for chunk in chunks:
            if (
                hasattr(chunk.message, "tool_call_chunks")
                and chunk.message.tool_call_chunks
            ):
                for tc in chunk.message.tool_call_chunks:
                    # Handle both ToolCallChunk objects and dictionaries
                    tc_id = tc.id if hasattr(tc, "id") else tc.get("id")
                    tc_name = tc.name if hasattr(tc, "name") else tc.get("name")
                    tc_args = tc.args if hasattr(tc, "args") else tc.get("args", "")

                    if tc_id not in tool_calls_by_id:
                        tool_calls_by_id[tc_id] = {
                            "id": tc_id,
                            "name": tc_name,
                            "args_str": "",
                        }

                    # Append args string
                    if tc_args:
                        tool_calls_by_id[tc_id]["args_str"] += tc_args

        # Convert the accumulated tool calls to the final format
        final_tool_calls = []
        for tc_id, tc_data in tool_calls_by_id.items():
            # Parse the arguments string
            try:
                tc_args_dict = (
                    json.loads(tc_data["args_str"]) if tc_data["args_str"] else {}
                )
                # Ensure args is a dict
                if not isinstance(tc_args_dict, dict):
                    tc_args_dict = {"value": tc_args_dict}
            except json.JSONDecodeError:
                # Fall back to the text parsing utility
                tc_args_dict = _parse_textual_tool_args(tc_data["args_str"])

            final_tool_calls.append(
                {
                    "id": tc_id,
                    "name": tc_data["name"],
                    "args": tc_args_dict,
                    "type": "function",
                }
            )

        # Set tool calls if any were found
        final_message = AIMessage(
            content=final_content,
            tool_calls=final_tool_calls if final_tool_calls else None,
        )

        # Set finish_reason if tool calls were found
        if final_tool_calls and "finish_reason" not in final_generation_info:
            final_generation_info["finish_reason"] = "tool_calls"

        # Add usage metadata
        total_tokens = prompt_tokens + completion_tokens
        if prompt_tokens > 0 or completion_tokens > 0:
            try:
                final_message.usage_metadata = UsageMetadata(
                    input_tokens=prompt_tokens,
                    output_tokens=completion_tokens,
                    total_tokens=total_tokens,
                )
            except Exception as e:
                logger.warning(f"Failed to set usage_metadata: {e}")

        # Create the final result
        result = ChatResult(
            generations=[
                ChatGeneration(
                    message=final_message, generation_info=final_generation_info
                )
            ],
            llm_output={
                "token_usage": {
                    "prompt_tokens": prompt_tokens,
                    "completion_tokens": completion_tokens,
                    "total_tokens": total_tokens,
                },
                "model_name": self.model_name,
            },
        )

        return result
