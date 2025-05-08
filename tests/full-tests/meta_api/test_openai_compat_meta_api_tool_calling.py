import pytest
import os
import dotenv
from openai import OpenAI, BadRequestError

dotenv.load_dotenv()
# API Configuration - credentials should be set as environment variables for tests
API_KEY = os.environ.get("META_API_KEY")
BASE_URL = os.environ.get("META_OAIC_API_BASE_URL", "https://api.llama.com/compat/v1/") #OpenAI compatibility endpoint (if ever needed for direct comparison)
MODEL_NAME = os.environ.get("META_MODEL_NAME", "Llama-4-Maverick-17B-128E-Instruct-FP8")  # Model name to use for testing
INTEGRATION_ENABLED = os.environ.get("INTEGRATION_ENABLED", "false").lower() == "true"

# Skip all tests in this file if the API key is not set OR if integration tests are not enabled
pytestmark = pytest.mark.skipif(
    not API_KEY or not INTEGRATION_ENABLED,
    reason="META_API_KEY environment variable not set OR INTEGRATION_ENABLED is not 'true'"
)

# Define a common tool structure for tests
common_tools = [
    {
        "type": "function",
        "function": {
            "name": "get_routing_decision",
            "description": "Selects the next destination or agent to act.",
            "parameters": {
                "type": "object",
                "properties": {
                    "destination": {
                        "type": "string",
                        "description": "The next agent or system to route to (e.g., EmailAgent, ScribeAgent, END).",
                    }
                },
                "required": ["destination"],
            },
        },
    }
]

@pytest.fixture(scope="module")
def client():
    if not API_KEY or not INTEGRATION_ENABLED:
        pytest.skip("META_API_KEY not set or INTEGRATION_ENABLED not true")
    return OpenAI(
        api_key=API_KEY,
        base_url=BASE_URL,
    )

def test_meta_tool_use_natural(client: OpenAI):
    """
    Test 1: Requesting a tool use naturally.
    Asserts that a structured tool_call is NOT made, and content is returned instead.
    """
    response = client.chat.completions.create(
        model=MODEL_NAME,
        messages=[
            {"role": "system", "content": "You are a helpful assistant that can make routing decisions."},
            {"role": "user", "content": "I need to send an email. What should be the next step? Use the get_routing_decision tool."},
        ],
        tools=common_tools,
    )
    assert response.choices is not None
    assert len(response.choices) > 0
    message = response.choices[0].message
    assert message.tool_calls is None, "Expected no structured tool_calls when tool_choice is 'auto' or not specified and model doesn't pick one"
    assert message.content is not None, "Expected content to be returned"
    # Based on observed behavior, the model might describe the tool call in text
    assert "get_routing_decision" in message.content.lower()

def test_meta_tool_choice_object_fails(client: OpenAI):
    """
    Test 2: Forcing tool use with tool_choice object format.
    Asserts that this raises a BadRequestError with the specific detail.
    """
    forced_tool_choice = {"type": "function", "function": {"name": "get_routing_decision"}}
    
    with pytest.raises(BadRequestError) as excinfo:
        client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": "You are a helpful assistant that must make a routing decision."},
                {"role": "user", "content": "Decide the route now."},
            ],
            tools=common_tools,
            tool_choice=forced_tool_choice,
        )
    
    assert excinfo.value.status_code == 400
    # The error response from Meta is a dict in the 'response.json()' field,
    # and openai library puts that into excinfo.value.body
    assert excinfo.value.body is not None
    assert "detail" in excinfo.value.body
    assert excinfo.value.body["detail"] == "tool_choice function object is not currently supported"

def test_meta_tool_choice_required_fails(client: OpenAI):
    """
    Test 3: Forcing tool use with tool_choice='required'.
    Asserts that this raises a BadRequestError with the specific detail.
    """
    with pytest.raises(BadRequestError) as excinfo:
        client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": "You are a helpful assistant that must make a routing decision using the provided tool."},
                {"role": "user", "content": "Decide the route now."},
            ],
            tools=common_tools,
            tool_choice="required",
        )

    assert excinfo.value.status_code == 400
    assert excinfo.value.body is not None
    assert "detail" in excinfo.value.body
    assert excinfo.value.body["detail"] == "tool_choice other than auto is not supported" 