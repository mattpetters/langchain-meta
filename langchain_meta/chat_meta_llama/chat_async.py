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
            completion_coro = self._astream_with_aggregation_and_retries(
                messages=messages,
                stop=stop,
                run_manager=run_manager,
                tools=tools,
                tool_choice=tool_choice,
                **kwargs,
            )
            return await self._aget_stream_results(completion_coro, run_manager)

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

        if run_manager:
            self._count_tokens(messages)
            pass

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

        # Buffer for aggregating a multi-chunk textual tool call
        # This is a simplified approach for now: we look for full textual tool calls in each chunk's content
        # A more robust solution would buffer across chunks if a textual call is split.

        current_tool_call_index = (
            0  # To assign index for AIMessageChunk tool_call_chunks
        )

        async for chunk in await active_client.chat.completions.create(**api_params):
            logger.debug(f"Llama API (async stream) Stream Chunk: {chunk.to_dict()}")
            chunk_dict = chunk.to_dict()
            content_str = ""

            # Enhanced content extraction for streaming chunks (existing logic)
            if hasattr(chunk, "completion_message") and chunk.completion_message:
                completion_msg = chunk.completion_message
                if hasattr(completion_msg, "content") and completion_msg.content:
                    content = completion_msg.content
                    if isinstance(content, dict) and "text" in content:
                        content_str = content["text"]
                    elif isinstance(content, str):
                        content_str = content
                if not content_str and hasattr(completion_msg, "to_dict"):
                    try:
                        msg_dict = completion_msg.to_dict()
                        if isinstance(msg_dict, dict) and "content" in msg_dict:
                            content_dict = msg_dict["content"]
                            if (
                                isinstance(content_dict, dict)
                                and "text" in content_dict
                            ):
                                content_str = content_dict["text"]
                            elif isinstance(content_dict, str):
                                content_str = content_dict
                    except (AttributeError, TypeError, KeyError):
                        pass
            if not content_str and chunk_dict:
                try:
                    if "completion_message" in chunk_dict:
                        comp_msg = chunk_dict["completion_message"]
                        if isinstance(comp_msg, dict) and "content" in comp_msg:
                            content = comp_msg["content"]
                            if isinstance(content, dict) and "text" in content:
                                content_str = content["text"]
                            elif isinstance(content, str):
                                content_str = content
                    if not content_str and "response_metadata" in chunk_dict:
                        # ... (existing deeper content extraction)
                        pass  # Assume existing deeper extraction handles this if needed
                except (KeyError, TypeError):
                    pass

            generation_info: Dict[str, Any] = {}  # Initialize for current chunk
            if hasattr(chunk, "stop_reason") and chunk.stop_reason:
                generation_info["finish_reason"] = chunk.stop_reason
            elif (
                hasattr(chunk, "finish_reason") and chunk.finish_reason
            ):  # Some Llama client versions might use this
                generation_info["finish_reason"] = chunk.finish_reason

            if chunk_dict.get("x_request_id"):
                generation_info["x_request_id"] = chunk_dict.get("x_request_id")
            if hasattr(chunk, "usage") and chunk.usage:
                generation_info["chunk_usage"] = chunk.usage.to_dict()

            # Process native tool calls from the API chunk first
            chunk_tool_calls_for_aimc_obj: List[ToolCallChunk] = []
            if (
                isinstance(chunk.message, AIMessageChunk)
                and chunk.message.tool_call_chunks
            ):
                for tc_chunk_dict in chunk.message.tool_call_chunks:
                    try:
                        # Construct ToolCallChunk objects
                        chunk_tool_calls_for_aimc_obj.append(
                            ToolCallChunk(
                                name=tc_chunk_dict.get("name"),
                                args=tc_chunk_dict.get("args", ""),
                                id=tc_chunk_dict.get("id"),
                                index=tc_chunk_dict.get("index"),
                            )
                        )
                    except Exception as e:
                        logger.warning(f"Could not construct ToolCallChunk: {e}")

            # Fallback: If no native tool calls in this chunk, try to parse from content_str
            if not chunk_tool_calls_for_aimc_obj and prepared_llm_tools and content_str:
                match = re.fullmatch(
                    r"\s*\[\s*([a-zA-Z0-9_]+)\s*(?:\(\s*(.*?)\s*\))?\s*\]\s*",
                    content_str,
                )
                if match:
                    tool_name_from_content = match.group(1)
                    args_str_from_content = match.group(
                        2
                    )  # This is a string, possibly JSON or empty

                    available_tool_names = [
                        t["function"]["name"]
                        for t in prepared_llm_tools
                        if isinstance(t, dict)
                        and "function" in t
                        and "name" in t["function"]
                    ]
                    if tool_name_from_content in available_tool_names:
                        logger.info(
                            f"Parsed textual tool call for '{tool_name_from_content}' from stream chunk content."
                        )
                        tool_call_id = str(uuid.uuid4())

                        # Use the new robust parser for args string
                        # However, for ToolCallChunk, args should be the string delta
                        # _parse_textual_tool_args(args_str_from_content) # We still call it to validate/log if needed

                        # For AIMessageChunk, 'args' should be the string delta.
                        # If args_str_from_content is None (e.g. [tool_name()]), pass empty string for args.
                        args_for_chunk_str = (
                            args_str_from_content
                            if args_str_from_content is not None
                            else ""
                        )

                        chunk_tool_calls_for_aimc_obj.append(
                            ToolCallChunk(
                                name=tool_name_from_content,
                                args=args_for_chunk_str,  # Use the original string delta
                                id=tool_call_id,
                                index=current_tool_call_index,  # Assign current index
                            )
                        )
                        current_tool_call_index += 1
                        content_str = ""  # Clear content as it's now a tool call
                        generation_info["finish_reason"] = (
                            "tool_calls"  # Mark finish reason
                        )
                    else:
                        logger.warning(
                            f"Textual tool call '{tool_name_from_content}' found in stream, but not in available tools."
                        )

            chunk = ChatGenerationChunk(
                message=AIMessageChunk(
                    content=content_str,
                    tool_call_chunks=chunk_tool_calls_for_aimc_obj,
                ),
                generation_info=generation_info if generation_info else None,
            )
            if run_manager and hasattr(run_manager, "on_llm_new_token"):
                await run_manager.on_llm_new_token(chunk.text, chunk=chunk)
            yield chunk

    async def _astream_with_aggregation_and_retries(
        self,
        messages: List[BaseMessage],
        run_manager: Optional[AsyncCallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> AsyncIterator[ChatGenerationChunk]:
        final_content = ""
        tool_deltas_by_index: Dict[int, Dict[str, Any]] = {}
        final_generation_info = {}

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
                tool_call_chunks_for_aimessagechunk = chunk.message.tool_call_chunks

            # Yield a new chunk that correctly represents the state *after* this incoming chunk
            # The AIMessageChunk content should be the delta (chunk.text), not aggregated across all prior chunks for yielding.
            # LangChain expects the AIMessageChunk.content to be the new token(s) in *this* chunk.
            yield ChatGenerationChunk(
                message=AIMessageChunk(
                    content=chunk.text,  # This is the delta for this chunk
                    tool_call_chunks=tool_call_chunks_for_aimessagechunk,
                ),
                generation_info=chunk.generation_info.copy()
                if chunk.generation_info
                else None,  # Pass along this chunk's specific info
            )
            # No further aggregation or yielding needed here; _aget_stream_results will handle the final aggregation.

    async def _aget_stream_results(
        self,
        completion_coro: AsyncIterator[ChatGenerationChunk],
        run_manager: Optional[AsyncCallbackManagerForLLMRun] = None,
    ) -> ChatResult:
        """Aggregates results from a stream of ChatGenerationChunks."""
        aggregated_content = ""

        # For tool calls, we need to aggregate potentially partial arguments for the same tool call ID.
        # LangChain's AIMessage expects fully formed tool_calls (List[Dict]), not ToolCallChunks.
        # We'll collect all ToolCallChunk data and then reconstruct full ToolCall dicts.

        # Store tool call parts by index, then by id.
        # Each entry in aggregated_tool_calls_parts will be a dict like:
        # { 'id': str, 'name': Optional[str], 'args_str_parts': List[str], 'type': str }
        aggregated_tool_call_parts_by_index: Dict[int, Dict[str, Any]] = {}

        final_generation_info_aggregated = {}
        last_chunk_for_finish_reason = None
        raw_response_dict_for_llm_output = None

        async for chunk in completion_coro:
            aggregated_content += chunk.text
            if chunk.generation_info:
                final_generation_info_aggregated.update(
                    chunk.generation_info
                )  # Overwrite with later info

            last_chunk_for_finish_reason = chunk  # Keep track of the last chunk
            if chunk.generation_info and isinstance(
                chunk.generation_info.get("response_metadata"), dict
            ):
                raw_response_dict_for_llm_output = chunk.generation_info.get(
                    "response_metadata"
                )
            elif chunk.generation_info:  # last resort for some dict
                raw_response_dict_for_llm_output = chunk.generation_info

            if chunk.message and chunk.message.tool_call_chunks:
                for tc_chunk in chunk.message.tool_call_chunks:
                    # tc_chunk is a dict: {'id': str, 'name': Optional[str], 'args': str (delta), 'index': int, 'type': Optional[str]}
                    idx = tc_chunk.get("index")
                    tc_id = tc_chunk.get("id")

                    if idx is not None:  # Streaming tool calls should have an index
                        if idx not in aggregated_tool_call_parts_by_index:
                            aggregated_tool_call_parts_by_index[idx] = {
                                "id": tc_id,
                                "name": tc_chunk.get("name"),
                                "args_str_parts": [],
                                "type": tc_chunk.get(
                                    "type", "function"
                                ),  # Default to function
                            }
                        # Update name if it wasn't set before (usually comes in first part)
                        if tc_chunk.get(
                            "name"
                        ) and not aggregated_tool_call_parts_by_index[idx].get("name"):
                            aggregated_tool_call_parts_by_index[idx]["name"] = (
                                tc_chunk.get("name")
                            )
                        if tc_id and not aggregated_tool_call_parts_by_index[idx].get(
                            "id"
                        ):  # Update ID if not set
                            aggregated_tool_call_parts_by_index[idx]["id"] = tc_id

                        args_delta = tc_chunk.get("args")
                        if isinstance(args_delta, str):
                            aggregated_tool_call_parts_by_index[idx][
                                "args_str_parts"
                            ].append(args_delta)

        # Reconstruct final tool_calls for AIMessage
        final_tool_calls_for_aimessage: List[Dict[str, Any]] = []
        for _idx, parts in sorted(
            aggregated_tool_call_parts_by_index.items()
        ):  # Process in order of index
            full_args_str = "".join(parts["args_str_parts"])
            parsed_args: Union[Dict, str]
            try:
                parsed_args = json.loads(full_args_str) if full_args_str else {}
            except json.JSONDecodeError:
                logger.warning(
                    f"Failed to parse aggregated tool call arguments for tool {parts.get('name')}. Using raw string."
                )
                parsed_args = full_args_str

            # Ensure parsed_args is a dict for the final AIMessage tool_call
            final_args_for_aimessage: Dict
            if isinstance(parsed_args, dict):
                final_args_for_aimessage = parsed_args
            else:  # If not a dict (e.g. was a string, or json.loads returned a non-dict)
                final_args_for_aimessage = {"value": str(parsed_args)}

            final_tool_calls_for_aimessage.append(
                {
                    "id": parts.get("id") or str(uuid.uuid4()),  # Ensure ID
                    "name": parts.get("name") or "unknown_tool",
                    "args": final_args_for_aimessage,
                    "type": parts.get("type", "function"),
                }
            )

        # Determine final finish_reason from the very last chunk or accumulated info
        if (
            last_chunk_for_finish_reason
            and last_chunk_for_finish_reason.generation_info
        ):
            final_generation_info_aggregated["finish_reason"] = (
                last_chunk_for_finish_reason.generation_info.get(
                    "finish_reason", "stop"
                )
            )
        elif "finish_reason" not in final_generation_info_aggregated:
            final_generation_info_aggregated["finish_reason"] = "stop"

        # Ensure 'duration' is present if other metrics like usage are (LangSmith might expect it)
        # This might be better handled by run_manager.on_llm_end if it calculates total duration.
        # For now, we'll ensure it's there if other usage data is.
        if (
            final_generation_info_aggregated.get("usage_metadata")
            and "duration" not in final_generation_info_aggregated
        ):
            # Placeholder if not set by _astream's final chunk info.
            # A more accurate duration would be from the start of _agenerate to now.
            final_generation_info_aggregated["duration"] = 0

        # Calculate final token usage (might be incomplete if API didn't send it)
        aggregated_token_usage = final_generation_info_aggregated.get(
            "usage_metadata", {}
        )
        prompt_tokens = aggregated_token_usage.get(
            "input_tokens", 0
        )  # Default to 0 if missing
        completion_tokens = aggregated_token_usage.get(
            "output_tokens", 0
        )  # Default to 0 if missing
        total_tokens = aggregated_token_usage.get(
            "total_tokens", prompt_tokens + completion_tokens
        )

        final_message = AIMessage(
            content=aggregated_content,
            tool_calls=final_tool_calls_for_aimessage,
        )
        # Ensure the message ID is set if not already, for tracking
        if not hasattr(final_message, "id") or not final_message.id:
            final_message.id = str(uuid.uuid4())

        # Attach usage metadata to the final message if available
        if aggregated_token_usage:
            try:
                final_message.usage_metadata = UsageMetadata(
                    input_tokens=prompt_tokens,
                    output_tokens=completion_tokens,
                    total_tokens=total_tokens,
                )
            except Exception as e:
                logger.warning(
                    f"Could not construct UsageMetadata for final message: {e}"
                )

        # Update generation_info for consistency before creating ChatResult
        final_generation_info_aggregated["token_usage"] = {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": total_tokens,
        }

        # ADDING LOGGING HERE
        logger.debug(
            f"_aget_stream_results: final_message ID: {final_message.id}"
        )  # Log ID
        logger.debug(
            f"_aget_stream_results: final_message.content: '{final_message.content}'"
        )
        logger.debug(
            f"_aget_stream_results: final_message.tool_calls: {final_message.tool_calls}"
        )
        logger.debug(
            f"_aget_stream_results: final_generation_info_aggregated: {final_generation_info_aggregated}"
        )

        final_result = ChatResult(
            generations=[
                ChatGeneration(
                    message=final_message,
                    generation_info=final_generation_info_aggregated,
                )
            ]
        )

        # Standardize llm_output for streaming
        llm_output_data = {
            "model_name": self.model_name,
            "token_usage": final_generation_info_aggregated["token_usage"],
            "system_fingerprint": final_generation_info_aggregated.get(
                "system_fingerprint"
            ),
            "request_id": final_generation_info_aggregated.get("x_request_id"),
            "finish_reason": final_generation_info_aggregated.get("finish_reason"),
            # Maybe include the last raw chunk dict if useful, or None for streaming
            "raw_response": raw_response_dict_for_llm_output,  # Might be None or last chunk's data
        }
        final_result.llm_output = llm_output_data

        if run_manager and hasattr(run_manager, "on_llm_end"):
            # Construct LLMResult for the callback
            generations_for_llm_result_stream: List[
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
                    final_result.generations,
                )
            ]
            llm_result_for_callback = LLMResult(
                generations=generations_for_llm_result_stream,
                llm_output=llm_output_data,
                run=None,
            )
            await run_manager.on_llm_end(llm_result_for_callback)

        return final_result
