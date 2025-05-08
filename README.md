# langchain-meta

This package contains the LangChain integration with ChatMetaLlama

## Installation

```bash
pip install -U langchain-meta
```

And you should configure credentials by setting the following environment variables:

* TODO: fill this out

## Chat Models

`ChatChatMetaLlama` class exposes chat models from ChatMetaLlama.

```python
from langchain_meta import ChatChatMetaLlama

llm = ChatChatMetaLlama()
llm.invoke("Sing a ballad of LangChain.")
```

## Embeddings

`ChatMetaLlamaEmbeddings` class exposes embeddings from ChatMetaLlama.

```python
from langchain_meta import ChatMetaLlamaEmbeddings

embeddings = ChatMetaLlamaEmbeddings()
embeddings.embed_query("What is the meaning of life?")
```

## LLMs
`ChatMetaLlamaLLM` class exposes LLMs from ChatMetaLlama.

```python
from langchain_meta import ChatMetaLlamaLLM

llm = ChatMetaLlamaLLM()
llm.invoke("The meaning of life is")
```
