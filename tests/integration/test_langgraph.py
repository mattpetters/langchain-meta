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
