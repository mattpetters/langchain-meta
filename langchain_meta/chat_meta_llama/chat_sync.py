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
)
from langchain_core.outputs import (
    ChatGeneration,
    ChatGenerationChunk,
    ChatResult,
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
            prepared_llm_tools = [
                _lc_tool_to_llama_tool_param(tool) for tool in tools_list_to_prepare
            ]
        else:
            logger.debug("_generate (sync): No effective tools to prepare.")

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

        if self.temperature is not None and "temperature" not in api_params:
            api_params["temperature"] = self.temperature
        # max_tokens from self.max_tokens is handled by _prepare_api_params if it's in **kwargs as max_completion_tokens
        # or if self.max_tokens is directly used by _prepare_api_params.
        # Here, we ensure it's passed if not already set by _prepare_api_params logic
        if (
            self.max_tokens is not None and "max_completion_tokens" not in api_params
        ):  # llama client uses max_completion_tokens
            if (
                "max_tokens" not in api_params
            ):  # Check if 'max_tokens' alias is also not there
                api_params["max_completion_tokens"] = self.max_tokens

        if (
            self.repetition_penalty is not None
            and "repetition_penalty" not in api_params
        ):
            api_params["repetition_penalty"] = self.repetition_penalty

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
        if self._client is None:
            raise ValueError("LlamaAPIClient not initialized.")

        # Use client from kwargs if provided (e.g. for testing), else use self._client
        active_client = kwargs.get("client") or self._client
        if not active_client:  # Should be caught by above, but defensive
            raise ValueError("Could not obtain an active LlamaAPIClient.")

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
                    f"_stream (sync): effective_tools_lc_input was not a list ({type(effective_tools_lc_input)}). Wrapping in list."
                )
                tools_list_to_prepare = [effective_tools_lc_input]
            prepared_llm_tools = [
                _lc_tool_to_llama_tool_param(tool) for tool in tools_list_to_prepare
            ]
        else:
            logger.debug("_stream (sync): No effective tools to prepare.")

        final_kwargs_for_prepare = kwargs.copy()
        final_kwargs_for_prepare.pop("tools", None)
        if tool_choice is not None:
            final_kwargs_for_prepare["tool_choice"] = tool_choice
        # tool_choice is implicitly handled by Llama API if tools are present or not for streaming.
        # If tool_choice needs to be explicitly passed for streaming, it would go into final_kwargs_for_prepare.

        api_params = self._prepare_api_params(
            messages,
            tools=prepared_llm_tools,
            stop=stop,
            stream=True,
            **final_kwargs_for_prepare,
        )
        logger.debug(f"Llama API (sync stream) Request: {api_params}")

        for chunk_result in active_client.chat.completions.create(**api_params):
            logger.debug(
                f"Llama API (sync stream) Stream Chunk: {chunk_result.to_dict()}"
            )
            chunk_dict = chunk_result.to_dict()
            content_str = ""

            # Enhanced content extraction for streaming chunks
            if (
                hasattr(chunk_result, "completion_message")
                and chunk_result.completion_message
            ):
                completion_msg = chunk_result.completion_message

                # Direct attribute access method - attempt 1
                if hasattr(completion_msg, "content") and completion_msg.content:
                    content = completion_msg.content
                    if isinstance(content, dict) and "text" in content:
                        content_str = content["text"]
                    elif isinstance(content, str):
                        content_str = content

                # Dictionary-based access - attempt 2
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

            # Traverse the full chunk dictionary if needed - attempt 3
            if not content_str and chunk_dict:
                try:
                    # Check completion_message in the chunk dictionary
                    if "completion_message" in chunk_dict:
                        comp_msg = chunk_dict["completion_message"]
                        if isinstance(comp_msg, dict) and "content" in comp_msg:
                            content = comp_msg["content"]
                            if isinstance(content, dict) and "text" in content:
                                content_str = content["text"]
                            elif isinstance(content, str):
                                content_str = content

                    # Look in response_metadata if it exists
                    if not content_str and "response_metadata" in chunk_dict:
                        response_meta = chunk_dict["response_metadata"]
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
                except (KeyError, TypeError):
                    pass

            generation_info = {}
            if hasattr(chunk_result, "stop_reason") and chunk_result.stop_reason:
                generation_info["finish_reason"] = chunk_result.stop_reason
            elif hasattr(chunk_result, "finish_reason") and chunk_result.finish_reason:
                generation_info["finish_reason"] = chunk_result.finish_reason

            if chunk_dict.get("x_request_id"):
                generation_info["x_request_id"] = chunk_dict.get("x_request_id")
            if hasattr(chunk_result, "usage") and chunk_result.usage:
                generation_info["chunk_usage"] = chunk_result.usage.to_dict()

            chunk_tool_calls = []
            if (
                hasattr(chunk_result, "completion_message")
                and chunk_result.completion_message
                and hasattr(chunk_result.completion_message, "tool_calls")
                and chunk_result.completion_message.tool_calls
            ):
                for idx, tc in enumerate(chunk_result.completion_message.tool_calls):
                    tool_call_data = {
                        "id": getattr(tc, "id", f"tc_{idx}"),
                        "type": "function",
                        "name": getattr(tc.function, "name", None)
                        if hasattr(tc, "function")
                        else None,
                        "args": getattr(tc.function, "arguments", "")
                        if hasattr(tc, "function")
                        else "",
                        "index": idx,
                    }
                    chunk_tool_calls.append(tool_call_data)

            chunk = ChatGenerationChunk(
                message=AIMessageChunk(
                    content=content_str or "",
                    tool_call_chunks=chunk_tool_calls if chunk_tool_calls else [],
                ),
                generation_info=generation_info if generation_info else None,
            )
            if run_manager:
                run_manager.on_llm_new_token(chunk.text, chunk=chunk)
            yield chunk
