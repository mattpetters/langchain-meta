"""Test ChatMetaLlama chat model integration."""

import os
from typing import Type

import pytest
from langchain_meta.chat_models import ChatMetaLlama
from langchain_tests.integration_tests import ChatModelIntegrationTests
from langchain.schema import BaseChatModel

# Fixture to skip tests if API key is not set
@pytest.fixture(scope="module")
def skip_if_no_api_key():
    if not (os.getenv("LLAMA_API_KEY") or os.getenv("META_API_KEY")):
        pytest.skip("Meta API key not set, skipping integration tests.")


@pytest.mark.usefixtures("skip_if_no_api_key")
class TestChatMetaLlamaIntegration(ChatModelIntegrationTests):
    @property
    def chat_model_class(self) -> Type[ChatMetaLlama]:
        return ChatMetaLlama

    @property
    def chat_model_params(self) -> dict:
        # Integration tests should use real credentials from environment variables
        # The ChatMetaLlama class attempts to read these if not provided directly.
        # We specify the model and temperature for consistency.
        params = {
            "model_name": os.getenv("LLAMA_TEST_MODEL", "Llama-3.3-8B-Instruct"), # Allow overriding model via env var
            "temperature": 0.7,
            "max_tokens": 100,
        }
        # Add API key and URL if they are explicitly set in env, otherwise let the class handle defaults/errors
        if key := os.getenv("LLAMA_API_KEY") or os.getenv("META_API_KEY"):
            params["llama_api_key"] = key
        if url := os.getenv("LLAMA_API_URL") or os.getenv("META_API_BASE_URL"):
            params["llama_api_url"] = url
        return params

    # Enable tests for optional features supported by Llama models
    @property
    def supports_image_inputs(self) -> bool:
        # Set to True if your specific model and implementation support image inputs
        # Llama 3.3 does not natively support image inputs in the base API yet
        return False

    @property
    def supports_video_inputs(self) -> bool:
        return False # Llama API does not support video inputs

    @property
    def supports_audio_inputs(self) -> bool:
        return False # Llama API does not support audio inputs

    @property
    def supports_tool_calling(self) -> bool:
        # Llama API supports tool calling
        return True

    @property
    def supports_structured_output(self) -> bool:
        # Assuming ChatMetaLlama implements with_structured_output
        # via function/tool calling or response_format
        return True # Llama API has response_format

    # Add any other overrides for optional capabilities if needed
    # See: https://python.langchain.com/v0.2/docs/contributing/how_to/integrations/standard_tests/#chat-models

    @pytest.mark.xfail(reason="Llama API client/API does not support tool_choice parameter.")
    def test_tool_choice(self, model: BaseChatModel) -> None:
        """Test that the model respects tool_choice.

        This test is expected to fail because the Llama API does not support the tool_choice parameter.
        """
        super().test_tool_choice(model)

    # Remove the duplicated test_tool_choice_expected_failure if present

    # Add any other overrides for optional capabilities if needed
    # See: https://python.langchain.com/v0.2/docs/contributing/how_to/integrations/standard_tests/#chat-models

    # Add any other overrides for optional capabilities if needed
    # See: https://python.langchain.com/v0.2/docs/contributing/how_to/integrations/standard_tests/#chat-models

    @pytest.mark.xfail(reason="Llama API client/API does not support tool_choice parameter.")
    def test_tool_choice(self, model: BaseChatModel) -> None:
        """Test that the model respects tool_choice.

        This test is expected to fail because the Llama API does not support the tool_choice parameter.
        """
        # This method is not applicable for the Llama API
        pytest.fail("Llama API does not support tool_choice parameter.")

    def test_tool_choice_expected_failure(self, model: BaseChatModel) -> None:
        """Test that the model respects tool_choice.

        This test is expected to fail because the Llama API does not support the tool_choice parameter.
        """
        # This method is not applicable for the Llama API
        pytest.fail("Llama API does not support tool_choice parameter.") 