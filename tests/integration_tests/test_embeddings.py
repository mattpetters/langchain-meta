"""Test ChatMetaLlama embeddings."""

from typing import Type

from langchain_meta.embeddings import ChatMetaLlamaEmbeddings
from langchain_tests.integration_tests import EmbeddingsIntegrationTests


class TestParrotLinkEmbeddingsIntegration(EmbeddingsIntegrationTests):
    @property
    def embeddings_class(self) -> Type[ChatMetaLlamaEmbeddings]:
        return ChatMetaLlamaEmbeddings

    @property
    def embedding_model_params(self) -> dict:
        return {"model": "nest-embed-001"}
