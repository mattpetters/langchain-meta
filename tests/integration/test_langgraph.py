# Integration test with langgraph
import os
import re
import asyncio  # Add asyncio for async test
from datetime import datetime
from typing import Annotated, List, Literal, Union

import pytest
from dotenv import load_dotenv
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, ToolMessage
from langchain_core.runnables import RunnableConfig
from langchain_core.tools import tool
from langchain_tavily import TavilySearch
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph, add_messages
from langgraph.prebuilt import ToolNode
from typing_extensions import TypedDict
import importlib
import typing

from langchain_meta.chat_models import ChatMetaLlama

load_dotenv()


# Boilerplate
class State(TypedDict):
    messages: Annotated[list, add_messages]


# Use @tool decorator
@tool
def get_current_time():
    """Get the current time."""
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


# Try importing Tavily if available - Define tools globally
try:
    tavily_tool = TavilySearch(max_results=2)
    tools_for_binding = [get_current_time, tavily_tool]
    print("Using Tavily search tool.")
except ImportError:
    tools_for_binding = [get_current_time]
    print("Tavily search tool not available, using only get_current_time.")


# Define the graph nodes
def call_model(state: State, config: RunnableConfig):
    messages = state["messages"]
    api_key = os.getenv("META_API_KEY")
    if not api_key:
        pytest.skip("META_API_KEY not set, skipping integration test.")

    # Use the specific Maverick model from the worklog
    # Use temperature 0 to make tool calling more deterministic for testing
    model = ChatMetaLlama(
        model_name="Llama-4-Maverick-17B-128E-Instruct-FP8",
        temperature=0.0,
        api_key=api_key,
    )

    # Bind tools
    model_with_tools = model.bind_tools(tools_for_binding)

    response = model_with_tools.invoke(messages, config=config)
    # We return a list, because this will get added to the existing list
    return {"messages": [response]}


async def acall_model(
    state: State, config: RunnableConfig
):  # Async version of call_model
    messages = state["messages"]
    api_key = os.getenv("META_API_KEY")
    if not api_key:
        pytest.skip("META_API_KEY not set, skipping integration test.")

    model = ChatMetaLlama(
        model_name="Llama-4-Maverick-17B-128E-Instruct-FP8",
        temperature=0.0,
        api_key=api_key,
    )
    model_with_tools = model.bind_tools(tools_for_binding)
    print(
        f"ASYNC acall_model: Invoking model with messages: {messages[-1].content if messages else 'No messages'}"
    )  # Log input
    response = await model_with_tools.ainvoke(messages, config=config)  # Use ainvoke
    # Log the response received by the node
    print(
        f"ASYNC acall_model: Received response ID: {getattr(response, 'id', None)}, content: '{getattr(response, 'content', None)}', tool_calls: {getattr(response, 'tool_calls', None)}, generation_info: {getattr(response, 'generation_info', None)}"
    )
    # Add debug logging for AIMessage properties in async tool node
    # (This is inside async_get_graph_final_state_messages and/or acall_model if present)
    for msg in [response]:
        print(f"[DEBUG] Message: {msg}")
        print(f"[DEBUG] tool_calls: {getattr(msg, 'tool_calls', None)}")
        print(f"[DEBUG] generation_info: {getattr(msg, 'generation_info', None)}")
    return {"messages": [response]}


# ToolNode expects tools argument
tool_node = ToolNode(tools_for_binding)


# Define conditional edges
def should_continue(state: State) -> Union[str, Literal["__end__"]]:
    messages = state["messages"]
    last_message = messages[-1]
    # If there are no tool calls, then we finish
    if not getattr(last_message, "tool_calls", []):
        return END
    # Otherwise if there are tool calls, we continue
    else:
        return "tools"


# Define and compile the graph globally (sync version for existing test)
graph_builder_sync = StateGraph(State)
graph_builder_sync.add_node("chatbot", call_model)
graph_builder_sync.add_node("tools", tool_node)
graph_builder_sync.add_edge(START, "chatbot")
graph_builder_sync.add_conditional_edges(
    "chatbot",
    should_continue,
)
graph_builder_sync.add_edge("tools", "chatbot")
memory_sync = MemorySaver()
app_sync = graph_builder_sync.compile(checkpointer=memory_sync)

# Define and compile the graph globally (async version for new test)
graph_builder_async = StateGraph(State)
graph_builder_async.add_node("chatbot", acall_model)  # Use async node
graph_builder_async.add_node("tools", tool_node)  # ToolNode can be reused
graph_builder_async.add_edge(START, "chatbot")
graph_builder_async.add_conditional_edges(
    "chatbot",
    should_continue,  # should_continue can be reused
)
graph_builder_async.add_edge("tools", "chatbot")
memory_async = MemorySaver()
app_async = graph_builder_async.compile(checkpointer=memory_async)


# Helper to stream graph updates and collect all messages from the final state
def stream_graph_and_collect_final_state_messages(input_text: str, thread_id: str):
    """Streams graph updates, prints them, and collects *all* messages from the final state."""
    print(f"--- Graph Stream Start for input: '{input_text}' (Thread: {thread_id}) ---")
    config: RunnableConfig = {"configurable": {"thread_id": thread_id}}
    final_state = None
    all_messages_in_final_state: List[BaseMessage] = []

    # Use stream to observe intermediate steps
    events = app_sync.stream(  # Use sync app
        {"messages": [HumanMessage(content=input_text)]}, config, stream_mode="values"
    )
    for i, event in enumerate(events):
        print(f"Event {i}: {event}")
        # Simplified printing for clarity
        if isinstance(event, dict):
            for key, value in event.items():
                print(f"  Event {i} - Key: {key}")
                # Special handling for message lists to show the last one
                if (
                    key == "chatbot"
                    and isinstance(value, dict)
                    and "messages" in value
                    and value["messages"]
                ):
                    last_msg = value["messages"][-1]
                    print(f"    Event {i} - Chatbot last message: {str(last_msg)[:70]}")
                    if hasattr(last_msg, "content"):
                        print(
                            f"    Event {i} - Chatbot last message content: {str(last_msg.content)[:70]}"
                        )
                    if hasattr(last_msg, "tool_calls") and last_msg.tool_calls:
                        print(
                            f"    Event {i} - Chatbot last message tool_calls: {str(last_msg.tool_calls)[:70]}"
                        )
                elif (
                    key == "tools"
                    and isinstance(value, dict)
                    and "messages" in value
                    and value["messages"]
                ):
                    tool_output = (
                        value  # Output of ToolNode is {'messages': [ToolMessage(...)]}
                    )
                    print(f"    Event {i} - Tools node output: {str(tool_output)[:70]}")
        final_state = event  # Keep track of the last state

    # Extract messages from the final state event
    if final_state and isinstance(final_state, dict) and "messages" in final_state:
        all_messages_in_final_state.extend(final_state["messages"])
    else:
        print(f"Warning: Final state event did not contain 'messages': {final_state}")

    print(
        f"--- Graph Stream End for input: '{input_text}'. Collected final state messages: {all_messages_in_final_state} ---"
    )
    return all_messages_in_final_state


async def async_get_graph_final_state_messages(
    input_text: str, thread_id: str
) -> List[BaseMessage]:
    """Invokes the graph asynchronously and collects all messages from the final state."""
    print(
        f"--- Async Graph Invoke Start for input: '{input_text}' (Thread: {thread_id}) ---"
    )
    config: RunnableConfig = {"configurable": {"thread_id": thread_id}}
    final_state = await app_async.ainvoke(  # Use async app and ainvoke
        {"messages": [HumanMessage(content=input_text)]}, config
    )
    all_messages_in_final_state: List[BaseMessage] = []
    if final_state and isinstance(final_state, dict) and "messages" in final_state:
        all_messages_in_final_state.extend(final_state["messages"])
    else:
        print(f"Warning: Final async state did not contain 'messages': {final_state}")
    print(
        f"--- Async Graph Invoke End for input: '{input_text}'. Collected final state messages: {len(all_messages_in_final_state)} ---"
    )
    # Add debug logging for AIMessage properties in async tool node
    # (This is inside async_get_graph_final_state_messages and/or acall_model if present)
    for msg in all_messages_in_final_state:
        print(f"[DEBUG] Message: {msg}")
        print(f"[DEBUG] tool_calls: {getattr(msg, 'tool_calls', None)}")
        print(f"[DEBUG] generation_info: {getattr(msg, 'generation_info', None)}")
    return all_messages_in_final_state


def import_tool_from_langchain_tools():
    try:
        langchain_tools = importlib.import_module("langchain.tools")
        return getattr(langchain_tools, "tool")
    except (ImportError, AttributeError):
        return None


@pytest.mark.integration
def test_chat_meta_llama_with_langchain_tools_decorator():
    """Test tool calling with @tool from langchain.tools (not langchain_core.tools)."""
    tool_decorator = import_tool_from_langchain_tools()
    if tool_decorator is None:
        pytest.skip("langchain.tools.tool not available; skipping test.")

    @tool_decorator
    def get_time_lc_tools():
        """Get the current time (langchain.tools)."""
        from datetime import datetime

        return datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # Use only this tool for the test
    tools = [get_time_lc_tools]

    # Build a minimal graph for this test
    def call_model_lc_tools(state: State, config: RunnableConfig):
        messages = state["messages"]
        api_key = os.getenv("META_API_KEY")
        if not api_key:
            pytest.skip("META_API_KEY not set, skipping integration test.")
        model = ChatMetaLlama(
            model_name="Llama-4-Maverick-17B-128E-Instruct-FP8",
            temperature=0.0,
            api_key=api_key,
        )
        model_with_tools = model.bind_tools(tools)
        response = model_with_tools.invoke(messages, config=config)
        return {"messages": [response]}

    graph_builder = StateGraph(State)
    graph_builder.add_node("chatbot", call_model_lc_tools)
    graph_builder.add_node("tools", ToolNode(tools))
    graph_builder.add_edge(START, "chatbot")
    graph_builder.add_conditional_edges("chatbot", should_continue)
    graph_builder.add_edge("tools", "chatbot")
    memory = MemorySaver()
    app = graph_builder.compile(checkpointer=memory)

    # Run the graph
    thread_id = "test-langchain-tools-thread"
    config = typing.cast(RunnableConfig, {"configurable": {"thread_id": thread_id}})
    events = app.stream(
        {"messages": [HumanMessage(content="What is the time?")]},
        config,
        stream_mode="values",
    )
    final_state = None
    all_messages = []
    for event in events:
        if isinstance(event, dict) and "messages" in event:
            all_messages.extend(event["messages"])
            final_state = event
    assert all_messages
    found_tool_message_with_time = False
    for msg in all_messages:
        if isinstance(msg, ToolMessage) and msg.name == "get_time_lc_tools":
            import re

            if re.search(r"\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}", str(msg.content)):
                found_tool_message_with_time = True
                break
    assert found_tool_message_with_time, (
        "No ToolMessage found with correctly formatted current time using langchain.tools.tool."
    )
    assert (
        all_messages
        and isinstance(all_messages[-1], AIMessage)
        and all_messages[-1].content
    ), "Final message was not an AIMessage with content (langchain.tools.tool)."


@pytest.mark.integration
def test_chat_meta_llama_integration():
    """Tests basic tool calling and response generation (SYNC)."""
    thread_id = "test-sync-thread-1"
    # config = {"configurable": {"thread_id": thread_id}} # Config is created in helper
    # memory.clear(config) # Removed clear call

    # Test user says "What is the time?"
    all_messages = stream_graph_and_collect_final_state_messages(
        "What is the time?", thread_id
    )
    assert all_messages

    # Check if *any* message in the final state is a ToolMessage with the correct time format
    found_tool_message_with_time = False
    for msg in all_messages:
        if isinstance(msg, ToolMessage) and msg.name == "get_current_time":
            # Check the content of the ToolMessage
            if re.search(r"\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}", str(msg.content)):
                found_tool_message_with_time = True
                break

    assert found_tool_message_with_time, (
        "SYNC: No ToolMessage found with correctly formatted current time."
    )

    # Check that the final message is an AIMessage with non-empty content
    assert (
        all_messages
        and isinstance(all_messages[-1], AIMessage)
        and all_messages[-1].content
    ), "SYNC: Final message was not an AIMessage with content."

    # Optional: Add a test for the Tavily tool if it was loaded
    if any(isinstance(t, TavilySearch) for t in tools_for_binding):
        print("\n--- Testing Tavily Search Tool (SYNC) ---")
        thread_id_tavily = "test-sync-thread-tavily"
        # config_tavily = {"configurable": {"thread_id": thread_id_tavily}} # Config created in helper
        # Clear memory again for the new thread
        # memory.clear(config_tavily) # Removed clear call
        tavily_messages = stream_graph_and_collect_final_state_messages(
            # Explicitly ask to use the tool
            "Use the tavily_search tool to find out what LangChain is.",
            thread_id_tavily,
        )
        assert tavily_messages
        # Check for AIMessage containing search results (simple check)
        assert any(
            isinstance(m, AIMessage) and "LangChain" in m.content
            for m in tavily_messages
        ), (
            "SYNC: Final response did not seem to contain Tavily search results about LangChain."
        )
        # Check for a ToolMessage from Tavily tool execution
        assert any(
            isinstance(m, ToolMessage) and m.name == "tavily_search"
            for m in tavily_messages
        ), "SYNC: No ToolMessage found for tavily_search in the final state messages."


@pytest.mark.integration
async def test_async_chat_meta_llama_integration():  # New async test
    """Tests basic tool calling and response generation (ASYNC)."""
    thread_id = "test-async-thread-1"
    # config = {"configurable": {"thread_id": thread_id}} # Config is created in helper
    # memory.clear(config) # Removed clear call

    # Test user says "What is the time?"
    all_messages = await async_get_graph_final_state_messages(
        "What is the time?", thread_id
    )
    assert all_messages

    # Check if *any* message in the final state is a ToolMessage with the correct time format
    found_tool_message_with_time = False
    for msg in all_messages:
        if isinstance(msg, ToolMessage) and msg.name == "get_current_time":
            # Check the content of the ToolMessage
            if re.search(r"\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}", str(msg.content)):
                found_tool_message_with_time = True
                break

    assert found_tool_message_with_time, (
        "ASYNC: No ToolMessage found with correctly formatted current time."
    )

    # Check that the final message is an AIMessage with non-empty content
    assert (
        all_messages
        and isinstance(all_messages[-1], AIMessage)
        and all_messages[-1].content
    ), "ASYNC: Final message was not an AIMessage with content."

    # Optional: Add a test for the Tavily tool if it was loaded
    if any(isinstance(t, TavilySearch) for t in tools_for_binding):
        print("\n--- Testing Tavily Search Tool (ASYNC) ---")
        thread_id_tavily = "test-async-thread-tavily"
        # config_tavily = {"configurable": {"thread_id": thread_id_tavily}} # Config created in helper
        # Clear memory again for the new thread
        # memory.clear(config_tavily) # Removed clear call
        tavily_messages = await async_get_graph_final_state_messages(
            # Explicitly ask to use the tool
            "Use the tavily_search tool to find out what LangChain is.",
            thread_id_tavily,
        )
        assert tavily_messages
        # Check for AIMessage containing search results (simple check)
        assert any(
            isinstance(m, AIMessage) and "LangChain" in m.content
            for m in tavily_messages
        ), (
            "ASYNC: Final response did not seem to contain Tavily search results about LangChain."
        )
        # Check for a ToolMessage from Tavily tool execution
        assert any(
            isinstance(m, ToolMessage) and m.name == "tavily_search"
            for m in tavily_messages
        ), "ASYNC: No ToolMessage found for tavily_search in the final state messages."


def import_interrupt_from_langgraph_types():
    try:
        langgraph_types = importlib.import_module("langgraph.types")
        return getattr(langgraph_types, "interrupt")
    except (ImportError, AttributeError):
        return None


def import_command_from_langgraph_types():
    try:
        langgraph_types = importlib.import_module("langgraph.types")
        return getattr(langgraph_types, "Command")
    except (ImportError, AttributeError):
        return None


@pytest.mark.integration
def test_human_assistance_tool_resume_with_command():
    """Test resuming a human_assistance interrupt with a Command(resume=...) and getting the human response."""
    interrupt = import_interrupt_from_langgraph_types()
    Command = import_command_from_langgraph_types()
    if interrupt is None or Command is None:
        pytest.skip(
            "langgraph.types.interrupt or Command not available; skipping test."
        )

    @tool
    def human_assistance(query: str) -> str:
        """Request assistance from a human."""
        human_response = interrupt({"query": query})
        return human_response["data"]

    tools = [get_current_time, human_assistance]

    def call_model_human_assist(state: State, config: RunnableConfig):
        messages = state["messages"]
        api_key = os.getenv("META_API_KEY")
        if not api_key:
            pytest.skip("META_API_KEY not set, skipping integration test.")
        model = ChatMetaLlama(
            model_name="Llama-4-Maverick-17B-128E-Instruct-FP8",
            temperature=0.0,
            api_key=api_key,
        )
        model_with_tools = model.bind_tools(tools)
        response = model_with_tools.invoke(messages, config=config)
        return {"messages": [response]}

    graph_builder = StateGraph(State)
    graph_builder.add_node("chatbot", call_model_human_assist)
    graph_builder.add_node("tools", ToolNode(tools))
    graph_builder.add_edge(START, "chatbot")
    graph_builder.add_conditional_edges("chatbot", should_continue)
    graph_builder.add_edge("tools", "chatbot")
    memory = MemorySaver()
    app = graph_builder.compile(checkpointer=memory)

    thread_id = "test-human-assistance-interrupt-resume"
    config = typing.cast(RunnableConfig, {"configurable": {"thread_id": thread_id}})
    input_text = "I need some expert guidance for building an AI agent. Could you request assistance for me?"
    # Step 1: Run until interrupt
    events = app.stream(
        {"messages": [HumanMessage(content=input_text)]}, config, stream_mode="values"
    )
    found_interrupt = False
    for event in events:
        if isinstance(event, dict) and any("interrupt" in k for k in event.keys()):
            found_interrupt = True
            break
    assert found_interrupt, "No interrupt was surfaced by the human_assistance tool."

    # Step 2: Resume with Command(resume={"data": ...})
    human_response = (
        "We, the experts are here to help! We'd recommend you check out LangGraph to build your agent."
        " It's much more reliable and extensible than simple autonomous agents."
    )
    human_command = Command(resume={"data": human_response})
    events = app.stream(human_command, config, stream_mode="values")
    found_human_response = False
    for event in events:
        if isinstance(event, dict) and "messages" in event:
            last_msg = event["messages"][-1]
            if hasattr(last_msg, "content") and human_response in str(last_msg.content):
                found_human_response = True
                break
    assert found_human_response, (
        "The human response was not found in the final message after resuming with Command."
    )


@pytest.mark.integration
def test_defensive_handling_of_malformed_tool_call():
    """Test that malformed tool calls (missing id or non-dict args) are handled defensively."""
    from langchain_meta.chat_meta_llama.serialization import _normalize_tool_call

    # Test case 1: Missing id, args is a string
    malformed1 = {"name": "dummy_tool", "args": "not_a_dict"}
    normalized1 = _normalize_tool_call(malformed1)
    assert "id" in normalized1
    assert isinstance(normalized1["id"], str)
    assert len(normalized1["id"]) > 0  # ID should be a non-empty string
    assert isinstance(normalized1["args"], dict)
    assert normalized1["args"]["value"] == "not_a_dict"
    assert normalized1["type"] == "function"

    # Test case 2: Empty id, args is None
    malformed2 = {"name": "dummy_tool", "id": "", "args": None}
    normalized2 = _normalize_tool_call(malformed2)
    assert "id" in normalized2
    assert isinstance(normalized2["id"], str)
    assert len(normalized2["id"]) > 0  # ID should be a non-empty string
    assert isinstance(normalized2["args"], dict)
    assert normalized2["args"] == {"value": "None"}

    # Test case 3: Missing name, valid args
    malformed3 = {"id": "tool123", "args": {"key": "value"}}
    normalized3 = _normalize_tool_call(malformed3)
    assert normalized3["name"] == "unknown_tool"
    assert normalized3["id"] == "tool123"
    assert normalized3["args"] == {"key": "value"}
