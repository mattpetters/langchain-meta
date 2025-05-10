"""LangChain <> Meta Llama API integration package."""

__version__ = "0.3.8"

from langchain_meta.chat_meta_llama.serialization import (
    serialize_message,
)
from langchain_meta.chat_models import (
    ChatMetaLlama,
)
from langchain_meta.utils import extract_json_response, meta_agent_factory

__all__ = [
    "ChatMetaLlama",
    "meta_agent_factory",
    "extract_json_response",
    "serialize_message",
]