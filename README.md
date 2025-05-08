# langchain-meta

Native integration between LangChain ecosystems and Meta's Llama API, providing direct access to Meta's powerful Llama models without OpenAI-compatible proxies.

## Installation

```bash
pip install -U langchain-meta
```

Set up your credentials with environment variables:

```bash
export META_API_KEY="your-api-key"
export META_API_BASE_URL="https://api.llama.com/v1"
export META_MODEL_NAME="Llama-4-Maverick-17B-128E-Instruct-FP8" 
# Optional, see list: https://llama.developer.meta.com/docs/api/models/
```

## Key Features

- **Direct Native API Access**: Connect to Meta's Llama models through their official API for full feature compatibility
- **Seamless Tool Calling**: Intelligent conversion between LangChain tool formats and Llama API requirements
- **Complete Message History Support**: Proper conversion of all LangChain message types
- **Multi-Agent System Compatibility**: Drop-in replacement for OpenAI in LangGraph workflows

## Chat Models

```python
from langchain_meta import ChatMetaLlama

llm = ChatMetaLlama()
llm.invoke("Who directed the movie The Social Network?")
```

## LangGraph & Multi-Agent Integration

The `ChatMetaLlama` class works seamlessly with LangGraph nodes and complex agent systems:

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

## Embeddings

```python
from langchain_meta import ChatMetaLlamaEmbeddings

embeddings = ChatMetaLlamaEmbeddings()
embeddings.embed_query("What is the meaning of life?")
```

## LLMs

```python
from langchain_meta import ChatMetaLlama

llm = ChatMetaLlama()
llm.invoke("The meaning of life is")
```

## Advanced Features

- **Streaming Support**: Full streaming implementation for both content and tool calls
- **Context Preservation**: Correctly handles the full conversation context in agent graphs
- **Error Resilience**: Robust handling of tool call parsing errors and response validation
- **Format Compatibility**: Support for structured output schemas like `RouteSchema`

By using the native Llama API integration, you can leverage Meta's models in complex LangChain architectures without the friction of format incompatibilities or API inconsistencies.
