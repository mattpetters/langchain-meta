# langchain-meta

Native integration between the [Meta Llama API](https://www.llama.com/products/llama-api/) 🦙 and the [LangChain/LangGraph ecosystem](https://www.langchain.com/), ⛓ providing fast hosted access to Meta's powerful Llama 4 models to power your Langgraph agents.
Fully implements [ChatModel interface](https://python.langchain.com/docs/concepts/chat_models/).

## Installation

```bash
pip install langchain-meta
```

Set up your credentials with environment variables:

```bash
export META_API_KEY="your-api-key"
export META_API_BASE_URL="https://api.llama.com/v1"
export META_MODEL_NAME="Llama-4-Maverick-17B-128E-Instruct-FP8"
# Optional, see list: https://llama.developer.meta.com/docs/api/models/
```

## Usage

### ChatMetaLlama

```python
from langchain_meta import ChatMetaLlama

# Initialize with API key and base URL
llm = ChatMetaLlama(
    model="Llama-4-Maverick-17B-128E-Instruct-FP8",
    api_key="your-meta-api-key",
    base_url="https://api.llama.com/v1/"
)

# Basic invocation
from langchain_core.messages import HumanMessage
response = llm.invoke([HumanMessage(content="Hello Llama!")])
print(response.content)
```

### LangSmith Integration

ChatMetaLlama is fully compatible with [LangSmith](https://smith.langchain.com/), providing comprehensive tracing and observability for your Meta LLM applications.

Key features of LangSmith integration:

- **Token Usage Tracking**: Get accurate input/output token counts for cost estimation
- **Request/Response Logging**: View full context of all prompts and completions
- **Tool Execution Tracing**: Monitor tool calls and their execution
- **Runtime Metrics**: Track latency and other performance metrics

To enable LangSmith tracing, set these environment variables:

```bash
export LANGSMITH_TRACING=true
export LANGSMITH_API_KEY="your-api-key"
export LANGSMITH_PROJECT="your-project-name"
```

See the [examples/langsmith_integration.py](./examples/langsmith_integration.py) script for a complete example of LangSmith integration.

### Utility Functions

#### meta_agent_factory

A utility to create LangChain runnables with Meta-specific configurations. Handles structured output and ensures streaming is disabled when needed for Meta API compatibility.

```python
from langchain_meta import meta_agent_factory, ChatMetaLlama
from langchain_core.tools import Tool
from pydantic import BaseModel

# Create LLM
llm = ChatMetaLlama(api_key="your-meta-api-key")

# Example with tools
tools = [Tool.from_function(func=lambda x: x, name="example", description="Example tool")]
agent = meta_agent_factory(
    llm=llm,
    tools=tools,
    system_prompt_text="You are a helpful assistant that uses tools.",
    disable_streaming=True
)

# Example with structured output
class ResponseSchema(BaseModel):
    answer: str
    confidence: float

structured_agent = meta_agent_factory(
    llm=llm,
    output_schema=ResponseSchema,
    system_prompt_text="Return structured answers with confidence scores."
)
```

#### extract_json_response

A robust utility to extract JSON from various response formats, handling direct JSON objects, code blocks with backticks, or JSON-like patterns in text.

```python
from langchain_meta import extract_json_response

# Parse various response formats
result = llm.invoke("Return a JSON with name and age")
parsed_json = extract_json_response(result.content)
```

## Key Features

- **Direct Native API Access**: Connect to Meta Llama models through their official API for full feature compatibility
- **Seamless Tool Calling**: Intelligent conversion between LangChain tool formats and Llama API requirements
- **Complete Message History Support**: Proper conversion of all LangChain message types
- **Multi-Agent System Compatibility**: Drop-in replacement for ChatOpenAI in LangGraph workflows
- **LangSmith Integration**: Full observability and tracing for debugging and monitoring

## Chat Models

```python
from langchain_meta import ChatMetaLlama

llm = ChatMetaLlama()
llm.invoke("Who directed the movie The Social Network?")
```

## LangGraph & Multi-Agent Integration

The `ChatMetaLlama` class works with LangGraph nodes and complex agent systems:

```python
from langchain_meta import ChatMetaLlama
from langchain_core.tools import tool

# Works with @tool decorations
@tool
def get_weather(location: str) -> str:
    """Get the current weather for a location."""
    return f"The weather in {location} is sunny."

# Create LLM with tools
llm = ChatMetaLlama(model="Llama-4-Maverick-17B-128E-Instruct-FP8")
llm_with_tools = llm.bind_tools([get_weather])

# Works in agent nodes and graph topologies
response = llm_with_tools.invoke("What's the weather in Seattle?")
```

## Advanced Features

- **Streaming Support**: Streaming implementation for both content and tool calls
- **Context Preservation**: Correctly handles the full conversation context in agent graphs
- **Error Resilience**: Robust handling of tool call parsing errors and response validation
- **Format Compatibility**: Support for structured output Pydantic objects
- **Observability**: Complete LangSmith integration for tracing and debugging

## Robust Tool Call Normalization

This module automatically normalizes all tool calls from Llama/Meta API:

- Ensures every tool call has a valid string `id` (generates one if missing/empty)
- Ensures `name` is a string (defaults to `"unknown_tool"` if missing)
- Ensures `args` is a dict (parses JSON if string, else wraps as `{"value": ...}`)
- Always sets `type` to `"function"`
- Logs a warning for any repair

**Best Practices:**

- Define your tool schemas as simply as possible (avoid advanced JSON Schema features).
- Provide clear prompt examples to the LLM for tool calling.
- Be aware of backend limitations (see [vLLM Issue #15236](https://github.com/vllm-project/vllm/issues/15236)).

Your code is robust to malformed tool calls, but clear schemas and prompts will maximize reliability.

## Defensive Tool Call Handling

LangChain Meta includes robust defensive mechanisms for handling malformed tool calls:

- Always ensures `tool_call_id` is a non-empty string (generates UUID if missing/empty)
- Always ensures `name` is a string (fallback to 'unknown_tool')
- Always ensures `args` is a dict (tries to parse string as JSON, else wraps as {'value': ...})
- Always sets `type` to 'function'
- Logs warnings for any repairs made

This makes the integration more resilient when working with models that may sometimes produce malformed tool calls.

## Contributing

We welcome contributions! Please see the [CONTRIBUTING.md](CONTRIBUTING.md) file for details.

## License

This project is licensed under the MIT License.

Llama 4, Llama AI API, etc trademarks belong to their respective owners (Meta)
I just made this to make my life easier and thought I'd share. 😊
