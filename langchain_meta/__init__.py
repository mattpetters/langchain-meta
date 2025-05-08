from importlib import metadata

from langchain_meta.chat_models import ChatChatMetaLlama
from langchain_meta.document_loaders import ChatMetaLlamaLoader
from langchain_meta.embeddings import ChatMetaLlamaEmbeddings
from langchain_meta.retrievers import ChatMetaLlamaRetriever
from langchain_meta.toolkits import ChatMetaLlamaToolkit
from langchain_meta.tools import ChatMetaLlamaTool
from langchain_meta.vectorstores import ChatMetaLlamaVectorStore

try:
    __version__ = metadata.version(__package__)
except metadata.PackageNotFoundError:
    # Case where package metadata is not available.
    __version__ = ""
del metadata  # optional, avoids polluting the results of dir(__package__)

__all__ = [
    "ChatChatMetaLlama",
    "ChatMetaLlamaVectorStore",
    "ChatMetaLlamaEmbeddings",
    "ChatMetaLlamaLoader",
    "ChatMetaLlamaRetriever",
    "ChatMetaLlamaToolkit",
    "ChatMetaLlamaTool",
    "__version__",
]
