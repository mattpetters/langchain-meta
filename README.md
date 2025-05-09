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

## Contributing

We welcome contributions! Please see the [CONTRIBUTING.md](CONTRIBUTING.md) file for details.

## License

This project is licensed under the MIT License.


Llama 4, Llama AI API, etc trademarks belong to their respective owners (Meta)
I just made this to make my life easier and thought I'd share. 😊