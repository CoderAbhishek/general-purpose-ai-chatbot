import pytest
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langgraph.errors import GraphRecursionError

from general_purpose_ai_chatbot.chatbot import graph as graph_module
from general_purpose_ai_chatbot.chatbot.tools import summariser as summariser_module


class FakeModel:
    """Stands in for ChatGroq: records what it was sent and returns scripted replies."""

    def __init__(self, replies):
        self.replies = list(replies)
        self.calls = []
        self.bound_tools = []

    def bind_tools(self, tools):
        self.bound_tools = tools
        return self

    def invoke(self, messages):
        self.calls.append(list(messages))
        # Repeat the last reply once the script runs out (used by the loop-limit test).
        reply = self.replies.pop(0) if len(self.replies) > 1 else self.replies[0]
        # Return a fresh copy, like a real model: the reducer assigns IDs to returned messages.
        return reply.model_copy()


def tool_call(name, args, call_id):
    return {"name": name, "args": args, "id": call_id}


def build_with(monkeypatch, replies):
    fake = FakeModel(replies)
    monkeypatch.setattr(graph_module, "create_chat_model", lambda: fake)
    return graph_module.build_graph(), fake


def invoke_and_trace(graph, text, config=None):
    visited = []
    final = None
    for chunk in graph.stream({"messages": [HumanMessage(text)]}, config, stream_mode=["updates", "values"]):
        mode, data = chunk
        if mode == "updates":
            visited.extend(data.keys())
        else:
            final = data
    return final, visited


# --- Structure -------------------------------------------------------------

def test_graph_structure(monkeypatch):
    graph, _ = build_with(monkeypatch, [AIMessage("hi")])
    drawn = graph.get_graph()
    assert {"llm", "tools"} <= set(drawn.nodes)
    edges = {(e.source, e.target) for e in drawn.edges}
    assert ("__start__", "llm") in edges
    assert ("llm", "tools") in edges
    assert ("llm", "__end__") in edges
    assert ("tools", "llm") in edges  # the loop back to the model
    assert ("tools", "__end__") not in edges


def test_same_tools_bound_to_model_and_tool_node(monkeypatch):
    _, fake = build_with(monkeypatch, [AIMessage("hi")])
    assert {t.name for t in fake.bound_tools} == {"calculator", "summarise_text", "tavily_search"}


# --- Execution paths (mocked model, real tools) ----------------------------

def test_direct_answer_skips_tools(monkeypatch):
    graph, fake = build_with(monkeypatch, [AIMessage("Hello!")])
    final, visited = invoke_and_trace(graph, "Hi there")

    assert visited == ["llm"]
    assert not any(isinstance(m, ToolMessage) for m in final["messages"])
    assert final["messages"][-1].content == "Hello!"
    assert len(fake.calls) == 1


def test_tool_loop_runs_tool_and_returns_to_model(monkeypatch):
    replies = [
        AIMessage("", tool_calls=[tool_call("calculator", {"a": 6, "b": 7, "operation": "multiply"}, "call_1")]),
        AIMessage("6 times 7 is 42."),
    ]
    graph, fake = build_with(monkeypatch, replies)
    final, visited = invoke_and_trace(graph, "What is 6 times 7?")

    assert visited == ["llm", "tools", "llm"]
    tool_messages = [m for m in final["messages"] if isinstance(m, ToolMessage)]
    assert len(tool_messages) == 1
    assert tool_messages[0].tool_call_id == "call_1"
    assert tool_messages[0].content == "42.0"
    # Second model call must include the tool result so the model can use it.
    assert any(isinstance(m, ToolMessage) for m in fake.calls[1])
    assert final["messages"][-1].content == "6 times 7 is 42."


def test_multiple_tool_calls_in_one_turn(monkeypatch):
    replies = [
        AIMessage("", tool_calls=[
            tool_call("calculator", {"a": 2, "b": 3, "operation": "add"}, "call_a"),
            tool_call("calculator", {"a": 2, "b": 3, "operation": "multiply"}, "call_b"),
        ]),
        AIMessage("Sum 5, product 6."),
    ]
    graph, _ = build_with(monkeypatch, replies)
    final, visited = invoke_and_trace(graph, "Add and multiply 2 and 3")

    assert visited == ["llm", "tools", "llm"]
    results = {m.tool_call_id: m.content for m in final["messages"] if isinstance(m, ToolMessage)}
    assert results == {"call_a": "5.0", "call_b": "6.0"}


# --- Failure paths -----------------------------------------------------------

def test_tool_exception_is_returned_not_crashed(monkeypatch):
    def groq_down():
        raise RuntimeError("Groq API unavailable")

    monkeypatch.setattr(summariser_module, "create_chat_model", groq_down)
    replies = [
        AIMessage("", tool_calls=[tool_call("summarise_text", {"text": "Long text here."}, "call_s")]),
        AIMessage("Sorry, summarising failed."),
    ]
    graph, _ = build_with(monkeypatch, replies)
    final, visited = invoke_and_trace(graph, "Summarise this")

    assert visited == ["llm", "tools", "llm"]
    tool_message = next(m for m in final["messages"] if isinstance(m, ToolMessage))
    assert "Groq API unavailable" in tool_message.content
    assert final["messages"][-1].content == "Sorry, summarising failed."


def test_unknown_tool_is_reported_to_model(monkeypatch):
    replies = [
        AIMessage("", tool_calls=[tool_call("weather", {"city": "Hyderabad"}, "call_w")]),
        AIMessage("I can't check the weather."),
    ]
    graph, _ = build_with(monkeypatch, replies)
    final, visited = invoke_and_trace(graph, "Weather in Hyderabad?")

    assert visited == ["llm", "tools", "llm"]
    tool_message = next(m for m in final["messages"] if isinstance(m, ToolMessage))
    assert "weather" in tool_message.content


def test_endless_tool_calls_stop_at_recursion_limit(monkeypatch):
    looping = AIMessage("", tool_calls=[tool_call("calculator", {"a": 1, "b": 1, "operation": "add"}, "loop")])
    graph, _ = build_with(monkeypatch, [looping])
    with pytest.raises(GraphRecursionError):
        graph.invoke({"messages": [HumanMessage("loop")]}, {"recursion_limit": 6})


# --- System prompt ---------------------------------------------------------

def test_system_prompt_sent_once_per_call_and_not_stored(monkeypatch):
    replies = [
        AIMessage("", tool_calls=[tool_call("calculator", {"a": 1, "b": 2, "operation": "add"}, "call_1")]),
        AIMessage("3"),
    ]
    graph, fake = build_with(monkeypatch, replies)
    final, _ = invoke_and_trace(graph, "1 + 2?")

    for sent in fake.calls:
        assert isinstance(sent[0], SystemMessage)
        assert sent[0].content == graph_module.SYSTEM_PROMPT
        assert sum(isinstance(m, SystemMessage) for m in sent) == 1
    assert not any(isinstance(m, SystemMessage) for m in final["messages"])
