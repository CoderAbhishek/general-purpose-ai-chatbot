import os

import pytest

from general_purpose_ai_chatbot.chatbot.model import create_chat_model
from general_purpose_ai_chatbot.chatbot.tools.calculator import calculator
from general_purpose_ai_chatbot.chatbot.tools.summariser import summarise_text
from general_purpose_ai_chatbot.chatbot.tools.web_search import MAX_RESULTS, create_web_search_tool

# Live tests are excluded by default (see addopts in pyproject.toml). Run with: uv run pytest -m live
pytestmark = [
    pytest.mark.live,
    pytest.mark.skipif(os.getenv("GROQ_API_KEY") == "test-api-key", reason="real GROQ_API_KEY not available"),
    pytest.mark.skipif(os.getenv("TAVILY_API_KEY") == "test-tavily-key", reason="real TAVILY_API_KEY not available"),
]

LONG_TEXT = (
    "LangGraph is a library for building stateful applications with large language models. "
    "It represents an application as a graph of nodes and edges, where each node is a step such as "
    "calling a model or running a tool. Conditional edges let the graph decide which node runs next, "
    "which makes it possible to build loops where a model calls tools until it has enough information. "
    "LangGraph also supports checkpointing, so a conversation can be resumed by thread ID, and streaming, "
    "so partial responses can be shown to the user while the model is still generating."
)


def test_summariser_live_groq_call():
    result = summarise_text.invoke({"text": LONG_TEXT})

    assert isinstance(result, str)
    assert result.strip()
    assert not result.startswith("Error:")
    assert len(result) < len(LONG_TEXT)


def test_groq_selects_calculator_for_arithmetic():
    # Real model + real tool schema: proves Groq accepts the schema and chooses the tool.
    model = create_chat_model().bind_tools([calculator])
    response = model.invoke("What is 1234 multiplied by 5678?")

    assert response.tool_calls, "model answered directly instead of calling a tool"
    call = response.tool_calls[0]
    assert call["name"] == "calculator"
    assert call["args"]["operation"] == "multiply"
    assert {float(call["args"]["a"]), float(call["args"]["b"])} == {1234.0, 5678.0}


def test_tavily_live_search_returns_results():
    # Real Tavily call: proves the key works and results come back with URLs.
    result = create_web_search_tool().invoke({"query": "LangGraph Python library"})

    assert "error" not in result, f"Tavily returned an error: {result}"
    assert 0 < len(result["results"]) <= MAX_RESULTS
    assert all(item["url"].startswith("http") for item in result["results"])


def test_groq_selects_web_search_for_current_events():
    # All three tools bound, as in the real chatbot: proves Groq picks search for current events
    # and can produce valid arguments for the larger Tavily schema.
    tools = [calculator, summarise_text, create_web_search_tool()]
    model = create_chat_model().bind_tools(tools)
    response = model.invoke("What are today's top technology news headlines?")

    assert response.tool_calls, "model answered directly instead of calling a tool"
    call = response.tool_calls[0]
    assert call["name"] == "tavily_search"
    assert call["args"]["query"].strip()
