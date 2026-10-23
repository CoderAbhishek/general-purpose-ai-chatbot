from typing import TypedDict, get_type_hints

from langchain_core.messages import AIMessage, AnyMessage, HumanMessage
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages

from general_purpose_ai_chatbot.chatbot.state import ChatState


# --- Reducer behaviour (add_messages called directly) ----------------------

def test_new_messages_are_appended():
    history = [HumanMessage("hi", id="1")]
    merged = add_messages(history, [AIMessage("hello", id="2")])
    assert [m.content for m in merged] == ["hi", "hello"]


def test_same_id_replaces_instead_of_duplicating():
    history = [HumanMessage("hi", id="1"), AIMessage("hello", id="2")]
    merged = add_messages(history, [AIMessage("hello again", id="2")])
    assert [m.content for m in merged] == ["hi", "hello again"]


def test_messages_without_id_get_one():
    merged = add_messages([], [HumanMessage("no id given")])
    assert merged[0].id is not None


def test_original_history_is_not_modified():
    history = [HumanMessage("hi", id="1")]
    add_messages(history, [AIMessage("hello", id="2")])
    assert len(history) == 1


# --- State definition --------------------------------------------------------

def test_chat_state_uses_add_messages_reducer():
    messages_hint = get_type_hints(ChatState, include_extras=True)["messages"]
    assert messages_hint.__metadata__ == (add_messages,)


# --- Reducer inside a real graph (no LLM) ------------------------------------

def reply_node(state):
    return {"messages": [AIMessage("hello from node")]}


def build_one_node_graph(state_type):
    builder = StateGraph(state_type)
    builder.add_node("reply", reply_node)
    builder.add_edge(START, "reply")
    builder.add_edge("reply", END)
    return builder.compile()


def test_graph_appends_node_output_to_history():
    result = build_one_node_graph(ChatState).invoke({"messages": [HumanMessage("hi")]})
    assert [m.content for m in result["messages"]] == ["hi", "hello from node"]


class StateWithoutReducer(TypedDict):
    messages: list[AnyMessage]


def test_without_reducer_history_is_overwritten():
    # Contrast case: same graph, no reducer. Shows why add_messages is needed.
    result = build_one_node_graph(StateWithoutReducer).invoke({"messages": [HumanMessage("hi")]})
    assert [m.content for m in result["messages"]] == ["hello from node"]
