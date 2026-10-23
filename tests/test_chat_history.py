from types import SimpleNamespace

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langgraph.checkpoint.memory import InMemorySaver

from fakes import StreamingFakeModel
from general_purpose_ai_chatbot.chatbot import graph as graph_module
from general_purpose_ai_chatbot.chatbot.graph import get_chat_history, stream_reply


class GraphWithMessages:
    """Minimal stand-in for a compiled graph: get_state returns the given messages."""

    def __init__(self, messages=None):
        self.values = {"messages": messages} if messages is not None else {}
        self.requested = None

    def get_state(self, config):
        self.requested = config
        return SimpleNamespace(values=self.values)


def ai_tool_call(*names):
    return AIMessage("", tool_calls=[{"name": n, "args": {}, "id": f"id_{i}"} for i, n in enumerate(names)])


# --- Unit: filtering and tool attribution ----------------------------------

def test_new_thread_has_empty_history():
    assert get_chat_history(GraphWithMessages(), "new") == []


def test_reads_the_requested_thread():
    graph = GraphWithMessages([])
    get_chat_history(graph, "abc")
    assert graph.requested == {"configurable": {"thread_id": "abc"}}


def test_plain_turn_has_no_tools():
    graph = GraphWithMessages([HumanMessage("Hi"), AIMessage("Hello!")])
    assert get_chat_history(graph, "t") == [
        {"role": "user", "text": "Hi", "tools": []},
        {"role": "assistant", "text": "Hello!", "tools": []},
    ]


def test_tool_messages_and_tool_requests_are_hidden_and_tools_attached():
    graph = GraphWithMessages([
        HumanMessage("6 x 7?"),
        ai_tool_call("calculator"),
        ToolMessage("42.0", tool_call_id="id_0"),
        AIMessage("It is 42."),
    ])
    assert get_chat_history(graph, "t") == [
        {"role": "user", "text": "6 x 7?", "tools": []},
        {"role": "assistant", "text": "It is 42.", "tools": ["calculator"]},
    ]


def test_same_tool_used_twice_is_listed_once():
    graph = GraphWithMessages([
        HumanMessage("Two sums"),
        ai_tool_call("calculator", "calculator"),
        ai_tool_call("calculator"),
        AIMessage("Done."),
    ])
    assert get_chat_history(graph, "t")[-1]["tools"] == ["calculator"]


def test_several_tools_kept_in_order_of_use():
    graph = GraphWithMessages([
        HumanMessage("Search then calculate"),
        ai_tool_call("tavily_search"),
        ai_tool_call("calculator"),
        AIMessage("Here you go."),
    ])
    assert get_chat_history(graph, "t")[-1]["tools"] == ["tavily_search", "calculator"]


def test_tools_reset_for_each_turn():
    graph = GraphWithMessages([
        HumanMessage("6 x 7?"),
        ai_tool_call("calculator"),
        AIMessage("42."),
        HumanMessage("Thanks"),
        AIMessage("You're welcome."),
    ])
    assert get_chat_history(graph, "t")[-1] == {"role": "assistant", "text": "You're welcome.", "tools": []}


def test_text_sent_with_a_tool_call_keeps_those_tools():
    # Known edge case: the model writes text AND requests a tool in one message.
    # The tools are attached to that early text; the final answer shows none.
    graph = GraphWithMessages([
        HumanMessage("6 x 7?"),
        AIMessage("Let me calculate that.", tool_calls=[{"name": "calculator", "args": {}, "id": "c1"}]),
        ToolMessage("42.0", tool_call_id="c1"),
        AIMessage("It is 42."),
    ])
    assert get_chat_history(graph, "t")[1:] == [
        {"role": "assistant", "text": "Let me calculate that.", "tools": ["calculator"]},
        {"role": "assistant", "text": "It is 42.", "tools": []},
    ]

# --- In-process integration: real graph and checkpointer, fake model -------

def test_history_from_a_real_graph_turn(monkeypatch):
    calculator_call = AIMessage(
        "", tool_calls=[{"name": "calculator", "args": {"a": 6, "b": 7, "operation": "multiply"}, "id": "c1"}]
    )
    fake = StreamingFakeModel(replies=[calculator_call, AIMessage("6 times 7 is 42")])
    monkeypatch.setattr(graph_module, "create_chat_model", lambda: fake)
    graph = graph_module.build_graph(checkpointer=InMemorySaver())

    list(stream_reply(graph, "What is 6 times 7?", "A"))

    assert get_chat_history(graph, "A") == [
        {"role": "user", "text": "What is 6 times 7?", "tools": []},
        {"role": "assistant", "text": "6 times 7 is 42", "tools": ["calculator"]},
    ]
