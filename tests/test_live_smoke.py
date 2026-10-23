import os

import pytest

from langchain_core.messages import HumanMessage, ToolMessage

from langgraph.checkpoint.memory import InMemorySaver

from general_purpose_ai_chatbot.chatbot.graph import build_graph, stream_reply
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


def test_graph_end_to_end_with_calculator():
    # Full loop with real Groq and the real calculator: question -> llm -> tools -> llm -> answer.
    result = build_graph().invoke({"messages": [HumanMessage("What is 1234 multiplied by 5678?")]})
    messages = result["messages"]

    assert any(isinstance(m, ToolMessage) for m in messages), "no tool was called"
    final = messages[-1]
    assert not final.tool_calls
    assert "7006652" in final.content.replace(",", "")


def test_live_streaming_yields_multiple_pieces():
    # Real Groq through the real graph: the reply arrives in several pieces, not one block.
    graph = build_graph(checkpointer=InMemorySaver())
    pieces = list(stream_reply(graph, "Explain in two sentences what a chatbot is.", "live-stream"))

    assert len(pieces) > 1
    assert "".join(pieces).strip()


def test_live_memory_and_thread_isolation():
    # Real Groq: the same thread remembers a fact; a new thread does not.
    graph = build_graph(checkpointer=InMemorySaver())
    list(stream_reply(graph, "My name is Abhishek. Please remember it.", "live-thread-1"))

    same_thread = "".join(stream_reply(graph, "What is my name?", "live-thread-1"))
    new_thread = "".join(stream_reply(graph, "What is my name?", "live-thread-2"))

    assert "abhishek" in same_thread.lower()
    assert "abhishek" not in new_thread.lower()
