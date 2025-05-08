# langchain-meta

This package contains a LangChain integration with ChatMetaLlama.
Wraps the official [Llama API Client/SDK from Meta](https://github.com/meta-llama/llama-api-python)

## Installation

```bash
pip install -U langchain-meta
```

And you should configure credentials by setting the following environment variables:

```bash
export META_API_KEY="your-api-key"
export META_NATIVE_API_BASE_URL="https://api.llama.com/v1"
export META_MODEL_NAME="Llama-4-Maverick-17B-128E-Instruct-FP8" 
# Optional, see list: https://llama.developer.meta.com/docs/api/models/
```

## Chat Models

`ChatMetaLlama` class exposes chat models from Meta Llama API.

```python
from langchain_meta import ChatMetaLlama

llm = ChatMetaLlama()
llm.invoke("Who directed the movie The Social Network?")
```

## Embeddings

`ChatMetaLlamaEmbeddings` class exposes embeddings from Meta Llama API.

```python
from langchain_meta import ChatMetaLlamaEmbeddings

embeddings = ChatMetaLlamaEmbeddings()
embeddings.embed_query("What is the meaning of life?")
```

## LLMs
`ChatMetaLlama` class exposes LLMs from Meta Llama API.

```python
from langchain_meta import ChatMetaLlama

llm = ChatMetaLlama()
llm.invoke("The meaning of life is")
```
