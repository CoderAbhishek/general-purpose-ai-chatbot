import os

import pytest

from general_purpose_ai_chatbot.chatbot.tools.summariser import summarise_text

# Live tests are excluded by default (see addopts in pyproject.toml). Run with: uv run pytest -m live
pytestmark = [
    pytest.mark.live,
    pytest.mark.skipif(os.getenv("GROQ_API_KEY") == "test-api-key", reason="real GROQ_API_KEY not available"),
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
