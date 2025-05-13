import json
import logging
import re
import uuid
from typing import (
    Any,
    Callable,
    Dict,
    List,
    Optional,
    Type,
    Union,
    cast,
)  # Added Callable and Optional here

from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)
from langchain_core.tools import BaseTool  # Added for _lc_tool_to_llama_tool_param
from llama_api_client.types import MessageParam  # Use MessageParam
from llama_api_client.types.chat import (
    completion_create_params,
)
from pydantic import BaseModel

logger = logging.getLogger(__name__)


def serialize_message(message: BaseMessage) -> Dict[str, Any]:
    """Serialize a LangChain message to a JSON-serializable dict."""
    result = {
        "type": message.__class__.__name__,
        "content": message.content
        if isinstance(message.content, str)
        else str(message.content),
        "additional_kwargs": {},
    }
    if message.additional_kwargs:
        for k, v in message.additional_kwargs.items():
            if k not in ["__pydantic_serializer__", "__pydantic_validator__"]:
                try:
                    json.dumps(v)
                    result["additional_kwargs"][k] = v
                except (TypeError, OverflowError):
                    result["additional_kwargs"][k] = str(v)
    if isinstance(message, AIMessage):
        if message.tool_calls:
            serialized_tool_calls = []
            for tc_item in message.tool_calls:
                tc_dict_curr: Dict[str, Any] = {}
                item_id = (
                    str(tc_item.get("id", uuid.uuid4()))
                    if isinstance(tc_item, dict)
                    else str(getattr(tc_item, "id", uuid.uuid4()))
                )
                item_name = (
                    str(tc_item.get("name", "unknown_tool"))
                    if isinstance(tc_item, dict)
                    else str(getattr(tc_item, "name", "unknown_tool"))
                )
                item_args = (
                    tc_item.get("args", {})
                    if isinstance(tc_item, dict)
                    else getattr(tc_item, "args", {})
                )

                tc_dict_curr["id"] = item_id
                tc_dict_curr["name"] = item_name

                if isinstance(item_args, dict):
                    try:
                        tc_dict_curr["args"] = json.dumps(item_args)
                    except (TypeError, OverflowError):
                        tc_dict_curr["args"] = str(item_args)
                elif item_args is not None:
                    tc_dict_curr["args"] = str(item_args)
                else:
                    tc_dict_curr["args"] = "{}"

                serialized_tool_calls.append(tc_dict_curr)
            result["tool_calls"] = serialized_tool_calls
        if message.additional_kwargs.get("function_call"):
            function_call = message.additional_kwargs["function_call"]
            if isinstance(function_call, dict) and "arguments" in function_call:
                try:
                    if not isinstance(function_call["arguments"], str):
                        function_call["arguments"] = json.dumps(
                            function_call["arguments"]
                        )
                    json.loads(function_call["arguments"])
                    result["function_call"] = function_call
                except (TypeError, OverflowError, json.JSONDecodeError):
                    func_call_copy = function_call.copy()
                    func_call_copy["arguments"] = str(function_call["arguments"])
                    result["function_call"] = func_call_copy
            else:
                result["function_call"] = str(function_call)
    elif isinstance(message, ToolMessage):
        result["tool_call_id"] = message.tool_call_id
        if not isinstance(message.content, str):
            try:
                if isinstance(message.content, (dict, list)):
                    result["content"] = json.dumps(message.content)
            except (TypeError, OverflowError):
                result["content"] = str(message.content)
    return result


def _lc_message_to_llama_message_param(
    message: BaseMessage,
) -> MessageParam:
    """Converts a LangChain BaseMessage to a Llama API MessageParam."""
    role: str
    content: Union[str, Dict[str, Any]]
    tool_calls: Optional[List[Dict[str, Any]]] = None
    tool_call_id: Optional[str] = None
    stop_reason: Optional[str] = None

    if isinstance(message, HumanMessage):
        role = "user"
        content_payload = message.content
    elif isinstance(message, AIMessage):
        role = "assistant"
        content_payload = message.content if message.content else ""
        if message.tool_calls and len(message.tool_calls) > 0:
            tool_calls = []
            for tc in message.tool_calls:
                args_val = tc.get("args")
                if isinstance(args_val, str):
                    try:
                        args_dict = json.loads(args_val)
                    except Exception:
                        args_dict = {"value": args_val}
                elif isinstance(args_val, dict):
                    args_dict = args_val
                else:
                    args_dict = {"value": str(args_val)}
                tool_calls.append(
                    {
                        "id": tc["id"],
                        "type": "function",
                        "function": {
                            "name": tc["name"],
                            "arguments": json.dumps(args_dict),
                        },
                    }
                )
            if (
                hasattr(message, "generation_info")
                and message.generation_info
                and "finish_reason" in message.generation_info
            ):
                if message.generation_info["finish_reason"] == "tool_calls":
                    stop_reason = "tool_calls"
            elif tool_calls:
                stop_reason = "tool_calls"
    elif isinstance(message, SystemMessage):
        role = "system"
        content_payload = message.content
    elif isinstance(message, ToolMessage):
        role = "tool"
        if isinstance(message.content, (list, dict)):
            content_payload = json.dumps(message.content)
        else:
            content_payload = str(message.content)
        tool_call_id = message.tool_call_id
    else:
        raise ValueError(f"Got unknown message type: {type(message)}")

    msg_dict: Dict[str, Any] = {
        "role": role,
        "content": content_payload,
    }

    if tool_calls:
        msg_dict["tool_calls"] = tool_calls
    if tool_call_id:
        msg_dict["tool_call_id"] = tool_call_id

    if role == "assistant" and tool_calls:
        msg_dict["content"] = ""

    if role == "assistant" and stop_reason:
        msg_dict["stop_reason"] = stop_reason

    return cast(MessageParam, msg_dict)


def _get_json_type_for_annotation(type_name: str) -> str:
    """Helper to convert Python type annotations to JSON schema types."""
    mapping = {
        "str": "string",
        "Text": "string",
        "string": "string",
        "int": "number",
        "float": "number",
        "complex": "number",
        "number": "number",
        "bool": "boolean",
        "boolean": "boolean",
        "list": "array",
        "tuple": "array",
        "array": "array",
        "List": "array",
        "Tuple": "array",
        "dict": "object",
        "Dict": "object",
        "mapping": "object",
        "Mapping": "object",
    }
    return mapping.get(type_name, "string")


def _convert_dict_tool(lc_tool: Any) -> dict:
    if (
        isinstance(lc_tool, dict)
        and "function" in lc_tool
        and isinstance(lc_tool["function"], dict)
    ):
        # Ensure strict: True is added if not present for consistency
        if "strict" not in lc_tool["function"]:
            lc_tool["function"]["strict"] = True
            logger.debug("Added strict: True to directly provided dict tool function.")
        return lc_tool
    raise ValueError("Not a dict tool suitable for Llama API direct use")


def _convert_pydantic_class_tool(
    lc_tool: Type[BaseModel],
) -> dict:
    name = getattr(lc_tool, "__name__", "UnnamedTool")
    description = getattr(lc_tool, "__doc__", "") or ""
    pydantic_schema = {}
    if hasattr(lc_tool, "model_json_schema") and callable(
        getattr(lc_tool, "model_json_schema")
    ):
        pydantic_schema = lc_tool.model_json_schema()
    elif hasattr(lc_tool, "schema") and callable(getattr(lc_tool, "schema")):
        pydantic_schema = lc_tool.schema()

    # Unwrap the Pydantic schema for Llama API
    llama_parameters: Dict[str, Any] = {}
    if isinstance(pydantic_schema, dict):
        if "properties" in pydantic_schema:
            llama_parameters["properties"] = pydantic_schema["properties"]
            # Special handling for include_domains/exclude_domains if generated with anyOf by Pydantic
            for field_name in ["include_domains", "exclude_domains"]:
                if (
                    field_name in llama_parameters["properties"]
                    and "anyOf" in llama_parameters["properties"][field_name]
                ):
                    original_field_def = llama_parameters["properties"][field_name]
                    has_array, has_null, array_schema_details = False, False, None
                    for sub_schema in original_field_def.get("anyOf", []):
                        if (
                            isinstance(sub_schema, dict)
                            and sub_schema.get("type") == "array"
                            and isinstance(sub_schema.get("items"), dict)
                            and sub_schema["items"].get("type") == "string"
                        ):
                            has_array, array_schema_details = True, sub_schema
                        if (
                            isinstance(sub_schema, dict)
                            and sub_schema.get("type") == "null"
                        ):
                            has_null = True
                    if has_array and has_null and array_schema_details:
                        new_field_def = {
                            "type": "array",
                            "items": array_schema_details["items"],
                        }
                        if "description" in original_field_def:
                            new_field_def["description"] = original_field_def[
                                "description"
                            ]
                        if "title" in original_field_def:
                            new_field_def["title"] = original_field_def["title"]
                        llama_parameters["properties"][field_name] = new_field_def
            # Special handling for Optional[bool] fields like include_images
            for field_name, field_def in list(
                llama_parameters.get("properties", {}).items()
            ):  # Iterate over a copy
                if isinstance(field_def, dict) and "anyOf" in field_def:
                    has_boolean, has_null = False, False
                    for sub_schema in field_def.get("anyOf", []):
                        if (
                            isinstance(sub_schema, dict)
                            and sub_schema.get("type") == "boolean"
                        ):
                            has_boolean = True
                        if (
                            isinstance(sub_schema, dict)
                            and sub_schema.get("type") == "null"
                        ):
                            has_null = True
                    if has_boolean and has_null:
                        new_simplified_def = {"type": "boolean"}
                        if "description" in field_def:
                            new_simplified_def["description"] = field_def["description"]
                        if "title" in field_def:
                            new_simplified_def["title"] = field_def["title"]
                        # If there was a default, it would be handled by Pydantic model or API if not sent
                        llama_parameters["properties"][field_name] = new_simplified_def
                        logger.debug(
                            f"Simplified Optional[bool] schema for field '{field_name}'."
                        )

            # Special handling for Optional[Literal] fields like search_depth, time_range, topic
            optional_literal_fields = ["search_depth", "time_range", "topic"]
            for field_name in optional_literal_fields:
                if (
                    field_name in llama_parameters["properties"]
                    and "anyOf" in llama_parameters["properties"][field_name]
                ):
                    original_field_def = llama_parameters["properties"][field_name]
                    has_enum_str, has_null = False, False
                    enum_schema_details = None
                    for sub_schema in original_field_def.get("anyOf", []):
                        if (
                            isinstance(sub_schema, dict)
                            and sub_schema.get("type") == "string"
                            and "enum" in sub_schema
                        ):
                            has_enum_str, enum_schema_details = True, sub_schema
                        if (
                            isinstance(sub_schema, dict)
                            and sub_schema.get("type") == "null"
                        ):
                            has_null = True
                    if has_enum_str and has_null and enum_schema_details:
                        new_field_def = {
                            "type": "string",
                            "enum": enum_schema_details["enum"],
                        }
                        if "description" in original_field_def:
                            new_field_def["description"] = original_field_def[
                                "description"
                            ]
                        if "title" in original_field_def:
                            new_field_def["title"] = original_field_def["title"]
                        llama_parameters["properties"][field_name] = new_field_def
                        logger.debug(
                            f"Simplified Optional[Literal] schema for field '{field_name}'."
                        )

        if "required" in pydantic_schema:
            llama_parameters["required"] = pydantic_schema["required"]
    else:
        logger.warning(
            f"Schema for {name} was not a dict: {type(pydantic_schema)}. Using empty parameters."
        )

    return {
        "type": "function",
        "function": {
            "name": name,
            "description": description,
            "parameters": llama_parameters,
            "strict": True,
        },
    }


def _convert_structured_tool(
    lc_tool: BaseTool,
) -> dict:  # Changed Any to BaseTool for clarity
    name = str(getattr(lc_tool, "name", "unnamed_tool"))
    description = str(getattr(lc_tool, "description", ""))
    pydantic_schema = {}
    if hasattr(lc_tool, "args_schema") and lc_tool.args_schema is not None:
        args_schema = lc_tool.args_schema
        if isinstance(args_schema, type) and issubclass(args_schema, BaseModel):
            pydantic_schema = args_schema.model_json_schema()
        elif isinstance(
            args_schema, dict
        ):  # If args_schema is already a dict, use it directly
            pydantic_schema = args_schema
        else:
            logger.warning(
                f"args_schema for {name} is not a Pydantic class or dict. Using empty schema."
            )

    # Unwrap the Pydantic schema for Llama API
    llama_parameters: Dict[str, Any] = {}
    if isinstance(pydantic_schema, dict):
        if "properties" in pydantic_schema:
            llama_parameters["properties"] = pydantic_schema["properties"]
            # Special handling for include_domains/exclude_domains if generated with anyOf by Pydantic
            for field_name in ["include_domains", "exclude_domains"]:
                if (
                    field_name in llama_parameters["properties"]
                    and "anyOf" in llama_parameters["properties"][field_name]
                ):
                    original_field_def = llama_parameters["properties"][field_name]
                    has_array, has_null, array_schema_details = False, False, None
                    for sub_schema in original_field_def.get("anyOf", []):
                        if (
                            isinstance(sub_schema, dict)
                            and sub_schema.get("type") == "array"
                            and isinstance(sub_schema.get("items"), dict)
                            and sub_schema["items"].get("type") == "string"
                        ):
                            has_array, array_schema_details = True, sub_schema
                        if (
                            isinstance(sub_schema, dict)
                            and sub_schema.get("type") == "null"
                        ):
                            has_null = True
                    if has_array and has_null and array_schema_details:
                        new_field_def = {
                            "type": "array",
                            "items": array_schema_details["items"],
                        }
                        if "description" in original_field_def:
                            new_field_def["description"] = original_field_def[
                                "description"
                            ]
                        if "title" in original_field_def:
                            new_field_def["title"] = original_field_def["title"]
                        llama_parameters["properties"][field_name] = new_field_def
            # Special handling for Optional[bool] fields like include_images (copied from _convert_pydantic_class_tool)
            for field_name, field_def in list(
                llama_parameters.get("properties", {}).items()
            ):  # Iterate over a copy
                if isinstance(field_def, dict) and "anyOf" in field_def:
                    has_boolean, has_null = False, False
                    for sub_schema in field_def.get("anyOf", []):
                        if (
                            isinstance(sub_schema, dict)
                            and sub_schema.get("type") == "boolean"
                        ):
                            has_boolean = True
                        if (
                            isinstance(sub_schema, dict)
                            and sub_schema.get("type") == "null"
                        ):
                            has_null = True
                    if has_boolean and has_null:
                        new_simplified_def = {"type": "boolean"}
                        if "description" in field_def:
                            new_simplified_def["description"] = field_def["description"]
                        if "title" in field_def:
                            new_simplified_def["title"] = field_def["title"]
                        llama_parameters["properties"][field_name] = new_simplified_def
                        logger.debug(
                            f"Simplified Optional[bool] schema for field '{field_name}' in structured tool."
                        )

            # Special handling for Optional[Literal] fields like search_depth, time_range, topic
            optional_literal_fields = ["search_depth", "time_range", "topic"]
            for field_name in optional_literal_fields:
                if (
                    field_name in llama_parameters["properties"]
                    and "anyOf" in llama_parameters["properties"][field_name]
                ):
                    original_field_def = llama_parameters["properties"][field_name]
                    has_enum_str, has_null = False, False
                    enum_schema_details = None
                    for sub_schema in original_field_def.get("anyOf", []):
                        if (
                            isinstance(sub_schema, dict)
                            and sub_schema.get("type") == "string"
                            and "enum" in sub_schema
                        ):
                            has_enum_str, enum_schema_details = True, sub_schema
                        if (
                            isinstance(sub_schema, dict)
                            and sub_schema.get("type") == "null"
                        ):
                            has_null = True
                    if has_enum_str and has_null and enum_schema_details:
                        new_field_def = {
                            "type": "string",
                            "enum": enum_schema_details["enum"],
                        }
                        if "description" in original_field_def:
                            new_field_def["description"] = original_field_def[
                                "description"
                            ]
                        if "title" in original_field_def:
                            new_field_def["title"] = original_field_def["title"]
                        llama_parameters["properties"][field_name] = new_field_def
                        logger.debug(
                            f"Simplified Optional[Literal] schema for field '{field_name}'."
                        )

        if "required" in pydantic_schema:
            llama_parameters["required"] = pydantic_schema["required"]
    else:
        logger.warning(
            f"Schema for {name} was not a dict or was empty. Using empty parameters."
        )

    function_def = {
        "name": name,
        "description": description,
        "parameters": llama_parameters,
        "strict": True,
    }
    # We removed additionalProperties earlier, ensure parameters itself is not empty for valid API call if no properties.
    # The Llama API examples show parameters: {} when no params, so ensure it's at least an empty dict.
    if not llama_parameters.get("properties") and not llama_parameters.get("required"):
        function_def[
            "parameters"
        ] = {}  # Ensure parameters is {} if no props/required, not just containing additionalProperties:false
    elif "properties" not in llama_parameters:  # if only required is present
        llama_parameters[
            "properties"
        ] = {}  # Llama might expect properties key even if empty if other keys like required are present
        function_def["parameters"] = llama_parameters

    return {"type": "function", "function": function_def}


def _convert_parse_method_tool(lc_tool: Any) -> dict:
    # This case is less common for Llama direct tools; often covered by StructuredTool
    if hasattr(lc_tool, "parse") and callable(getattr(lc_tool, "parse")):
        name = getattr(
            lc_tool, "name", getattr(lc_tool, "__class__", type(lc_tool)).__name__
        )
        description = (
            getattr(lc_tool, "__doc__", "") or f"Tool that parses input for {name}"
        )
        schema = getattr(lc_tool, "schema", None)
        parameters = (
            schema()
            if callable(schema)
            else (
                schema
                if isinstance(schema, dict)
                else {"type": "object", "properties": {}}
            )
        )
        if isinstance(parameters, dict) and parameters.get("type") == "object":
            parameters["additionalProperties"] = False
        return {
            "type": "function",
            "function": {
                "name": name,
                "description": description,
                "parameters": parameters,
                "strict": True,
            },
        }
    raise ValueError("Not a parse-method tool or schema extraction failed")


def _convert_route_schema_tool(lc_tool: Any) -> dict:
    # Specific handling for a tool named "RouteSchema" or "route_schema"
    # This seems very application-specific, ensure it's truly generic or handled by specific tool logic.
    if hasattr(lc_tool, "name") and getattr(lc_tool, "name") in [
        "RouteSchema",
        "route_schema",
    ]:
        # This enum should ideally be dynamic or part of the tool's definition
        enum_values = [
            "EmailAgent",
            "ScribeAgent",
            "TimeKeeperAgent",
            "GeneralAgent",
            "END",
            "__end__",
        ]
        return {
            "type": "function",
            "function": {
                "name": getattr(lc_tool, "name"),
                "description": "Route to the next agent",
                "parameters": {
                    "type": "object",
                    "properties": {"next": {"type": "string", "enum": enum_values}},
                    "required": ["next"],
                },
                "strict": True,  # Added strict here as well
            },
        }
    raise ValueError("Not a RouteSchema tool by name convention")


def _create_minimal_tool(lc_tool: Any) -> dict:
    name = str(getattr(lc_tool, "name", type(lc_tool).__name__))
    description = str(getattr(lc_tool, "description", ""))
    logger.error(
        f"Could not convert tool {name} ({type(lc_tool)}) to Llama API format. Creating fallback."
    )
    parameters = {"type": "object", "properties": {}, "additionalProperties": False}
    return {
        "type": "function",
        "function": {
            "name": name,
            "description": description,
            "parameters": parameters,
            "strict": True,
        },
    }


def _lc_tool_to_llama_tool_param(
    lc_tool: Union[Dict[str, Any], Type[BaseModel], Callable, BaseTool],
) -> completion_create_params.Tool:
    """Convert LangChain tool to Llama API format using a refined dispatcher pattern."""
    # Check for direct Llama API dict format first (common for bind_tools with pre-formatted dicts)
    if (
        isinstance(lc_tool, dict)
        and "function" in lc_tool
        and isinstance(lc_tool["function"], dict)
    ):
        try:
            return _convert_dict_tool(lc_tool)  # type: ignore
        except ValueError:
            pass  # Fall through if not perfectly matching

    # Pydantic model class (e.g., MyToolSchema(BaseModel))
    if isinstance(lc_tool, type) and issubclass(lc_tool, BaseModel):
        return _convert_pydantic_class_tool(lc_tool)  # type: ignore

    # LangChain BaseTool (includes StructuredTool, Tool)
    if isinstance(lc_tool, BaseTool):
        # StructuredTool and Tool have .name, .description, and often .args_schema or .schema_
        return _convert_structured_tool(lc_tool)  # type: ignore

    # LangChain tools defined via @tool decorator (becomes a Runnable with .name, .description, .args_schema)
    # This case is often caught by BaseTool isinstance check if @tool produces a BaseTool subclass.
    # If it produces a plain Callable wrapped in some other Runnable, more checks might be needed.
    # For now, relying on BaseTool check or falling through.

    # If it has a parse method (less common for direct Llama tools but for completeness)
    if hasattr(lc_tool, "parse") and callable(getattr(lc_tool, "parse")):
        try:
            return _convert_parse_method_tool(lc_tool)  # type: ignore
        except ValueError:
            pass

    # Application-specific named tool like RouteSchema (if this pattern is intended to be general)
    if hasattr(lc_tool, "name") and getattr(lc_tool, "name") in [
        "RouteSchema",
        "route_schema",
    ]:
        try:
            return _convert_route_schema_tool(lc_tool)  # type: ignore
        except ValueError:
            pass

    # Fallback for any other type or if above conversions failed subtly
    return _create_minimal_tool(lc_tool)  # type: ignore


def _normalize_tool_call(tc: dict) -> dict:
    """
    Defensive normalization for tool call dicts:
    - Ensures 'id' is a non-empty string (generates uuid if missing/empty)
    - Ensures 'name' is a string (fallback to 'unknown_tool')
    - Ensures 'args' is a dict (parses string as JSON, else wraps as {'value': ...})
    - Always sets 'type' to 'function'
    - Logs a warning for any repair
    """
    logger = logging.getLogger(__name__)
    tool_call = dict(tc)  # shallow copy

    # ID
    tool_call_id = tool_call.get("id")
    if (
        not tool_call_id
        or not isinstance(tool_call_id, str)
        or not tool_call_id.strip()
    ):
        tool_call_id = str(uuid.uuid4())
        logger.warning(f"Tool call missing or invalid id. Generated: {tool_call_id}")
    tool_call["id"] = tool_call_id

    # Name
    name = tool_call.get("name")
    if not name or not isinstance(name, str):
        logger.warning(f"Tool call missing or invalid name. Using 'unknown_tool'.")
        name = "unknown_tool"
    tool_call["name"] = name

    # Args
    args_val = tc.get("args")
    if isinstance(args_val, str):
        try:
            args_dict = json.loads(args_val)
        except Exception:
            args_dict = {"value": args_val}
    elif isinstance(args_val, dict):
        args_dict = args_val
    else:
        args_dict = {"value": str(args_val)}
    tool_call["args"] = args_dict

    # Type
    tool_call["type"] = "function"

    return tool_call


def _parse_textual_tool_args(args_str: Optional[str]) -> Dict[str, Any]:
    """
    Parses a string of arguments like 'key="value", key2=value2' into a dict.
    Handles JSON-like structures if possible, otherwise falls back to regex parsing.
    """
    if not args_str or not args_str.strip():
        return {}

    # Attempt to parse as JSON first, as it's the most robust
    try:
        # Ensure outer braces for valid JSON object if it's just key:value pairs
        # This handles cases like '{"key": "value"}' or even 'key: "value"' if it's valid enough
        potential_json_str = args_str
        if not potential_json_str.startswith("{") or not potential_json_str.endswith(
            "}"
        ):
            # Basic check if it looks like a raw string needing to be quoted for a single arg tool
            if (
                ":" not in potential_json_str
                and "=" not in potential_json_str
                and '"' not in potential_json_str
            ):
                # It might be a single unquoted string for a tool that takes one arg named e.g. "query" or "input"
                # We can't know the arg name here, so we'll wrap it with a default key like "value" or let regex handle it
                pass  # Let regex try or handle as single value if JSON fails

        # More robust JSON parsing attempt
        try:
            loaded_args = json.loads(potential_json_str)
            if isinstance(loaded_args, dict):
                return loaded_args
        except json.JSONDecodeError:
            # If it's not a valid JSON object string, try adding braces
            if not potential_json_str.startswith("{"):
                potential_json_str = "{" + potential_json_str
            if not potential_json_str.endswith("}"):
                potential_json_str = potential_json_str + "}"

            # Try to make key=value into "key":"value"
            # This is a simplified attempt and might not cover all edge cases for non-standard formats
            # Regex to find key=value or key="value" pairs.
            # It tries to match keys (alphanumeric, underscores) and values (quoted or unquoted).
            # Handles basic cases, but complex nested structures or unusual characters in unquoted values might fail.

            # Attempt 1: Try to parse key="value" or key='value' or key=value (unquoted)
            # This regex handles simple key=value, key="value", key='value'
            # It's hard to make one regex perfect for all malformed "JSON-like" strings.
            # key\s*=\s*(?:\"(.*?)\"|'(.*?)'|([^,\"'\s]+))

            # For `query="What is LangChain"` the goal is `{"query": "What is LangChain"}`

            # Simpler approach: if it's just one `key="value"`:
            single_arg_match = re.match(
                r"^\s*([a-zA-Z_][a-zA-Z0-9_]*)\s*=\s*\"(.*?)\"\s*$", args_str
            )
            if single_arg_match:
                return {single_arg_match.group(1): single_arg_match.group(2)}

            try:
                # This is a very basic attempt to convert Python-like dict string to JSON
                # It assumes keys are unquoted or quoted with single quotes, and strings use double quotes
                # Not robust for complex cases.
                python_like_dict_str = args_str
                # Ensure keys are double-quoted
                python_like_dict_str = re.sub(
                    r"([{\s,])([a-zA-Z_][a-zA-Z0-9_]*)\s*:",
                    r'\1"\2":',
                    python_like_dict_str,
                )
                # Ensure single-quoted strings become double-quoted
                python_like_dict_str = re.sub(r"'", r'"', python_like_dict_str)

                # If it doesn't look like a dict, and json.loads failed, wrap it
                if not python_like_dict_str.strip().startswith("{"):
                    python_like_dict_str = "{" + python_like_dict_str + "}"

                loaded_args = json.loads(python_like_dict_str)
                if isinstance(loaded_args, dict):
                    return loaded_args

            except json.JSONDecodeError:
                pass  # Fall through to regex or default

    except json.JSONDecodeError:
        # Fallback to regex for simple key="value" or key=value cases if JSON fails completely
        pass

    # Fallback regex for key="value", key='value', key=value (unquoted, simple)
    # This is a simplified regex and might not capture all desired formats perfectly,
    # especially with complex values or multiple arguments not well-separated.
    args = {}
    # Regex to find key=value pairs where value can be quoted or unquoted.
    # It captures: 1=key, 2=double-quoted value, 3=single-quoted value, 4=unquoted value
    # SIMPLIFIED REGEX TO GET PAST SYNTAX ERROR
    pattern = re.compile(
        r"([a-zA-Z_][a-zA-Z0-9_]*)\s*=\s*\"(.*?)\""
    )  # Focus on key="value"
    for match in pattern.finditer(args_str):
        key = match.group(1)
        value = match.group(2)  # Only one value group now
        # Try to convert to int/float/bool if applicable, otherwise keep as string
        if value.lower() == "true":
            args[key] = True

    if not args and args_str:  # If regex found nothing but there was an args_str
        # This could be a single string argument for a tool that expects e.g. a query.
        # We can't know the schema here, so we return it with a default key 'value'
        # or the user of this function has to be aware.
        # For the specific case of `query=\"What is LangChain\"` failing with the above,
        # this won't help if it's not parsed by the regex.

        # REMOVING PROBLEMATIC FALLBACK LOGIC TO ISOLATE SYNTAX ERROR
        # # One last attempt for `key=\"value\"` where key might have spaces (not ideal for keys)
        # # or for just a single value that should be the query
        # # The provided `query=\"What is LangChain\"` should be caught by the regex.
        # # The issue is that the value has a quote inside: `\"What is LangChain\"`
        # # Let's refine the regex slightly for quoted values that might contain escaped quotes
        # # This is hard without knowing the exact escaping rules.
        #
        # # If all else fails and the string contains 'query=', assume it's the primary argument for Tavily
        # # Corrected quote escaping
        # query_match = re.search(r'query=(?:"(.*?)"|'(.*?)'|([^,\s]+))', args_str, re.IGNORECASE)
        # if query_match:
        #     query_val = next((g for g in query_match.groups() if g is not None), None)
        #     if query_val:
        #         return {"query": query_val}

        # Default if no parsing worked but string is not empty
        return {"value": args_str}

    return args
