"""Test chat model integration."""

from unittest.mock import MagicMock, patch
import pytest
from langchain_meta import ChatMetaLlama
from langchain_core.messages import HumanMessage


@pytest.fixture
def mock_llama_client():
    with patch("langchain_meta.chat_models.LlamaAPIClient") as mock_client:
        yield mock_client


def test_chat_meta_llama_initialization(mock_llama_client):
    llm = ChatMetaLlama(api_key="test-key")
    assert llm.model_name in llm._identifying_params


@pytest.mark.asyncio
async def test_agenerate_method(mock_llama_client):
    mock_instance = mock_llama_client.return_value
    mock_instance.chat.completions.create.return_value.completion_message.content = (
        "Test response"
    )

    llm = ChatMetaLlama(api_key="test-key")
    messages = [HumanMessage(content="Hello")]
    result = await llm._agenerate(messages)

    assert len(result.generations) == 1
    assert "Test response" in result.generations[0].message.content
