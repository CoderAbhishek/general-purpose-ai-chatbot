import json

import pytest
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, AIMessageChunk, HumanMessage, ToolMessage
from langchain_core.outputs import ChatGeneration, ChatGenerationChunk, ChatResult
from langgraph.checkpoint.memory import InMemorySaver
from pydantic import Field

from general_purpose_ai_chatbot.chatbot import graph as graph_module
from general_purpose_ai_chatbot.chatbot.graph import stream_reply


class StreamingFakeModel(BaseChatModel):
    """A real LangChain chat model with scripted replies.

    Unlike a plain fake, it streams text word by word through LangChain's callbacks,
    exactly the path stream_mode="messages" listens to. Tool-call replies are sent as
    tool-call chunks, as a real model would.
    """

    replies: list = Field(default_factory=list)
    calls: list = Field(default_factory=list)

    @property
    def _llm_type(self) -> str:
        return "streaming-fake"

    def bind_tools(self, tools, **kwargs):
        return self

    def _next_reply(self, messages):
        self.calls.append(list(messages))
        return self.replies.pop(0) if len(self.replies) > 1 else self.replies[0]

    def _generate(self, messages, stop=None, run_manager=None, **kwargs):
        return ChatResult(generations=[ChatGeneration(message=self._next_reply(messages).model_copy())])

    def _stream(self, messages, stop=None, run_manager=None, **kwargs):
        reply = self._next_reply(messages)
        if reply.tool_calls:
            chunks = [
                {"name": c["name"], "args": json.dumps(c["args"]), "id": c["id"], "index": i}
                for i, c in enumerate(reply.tool_calls)
            ]
            yield ChatGenerationChunk(message=AIMessageChunk(content="", tool_call_chunks=chunks))
            return
        for i, word in enumerate(reply.content.split(" ")):
            text = word if i == 0 else " " + word
            chunk = ChatGenerationChunk(message=AIMessageChunk(content=text))
            if run_manager:
                run_manager.on_llm_new_token(text, chunk=chunk)
            yield chunk


def use_fake(monkeypatch, replies):
    fake = StreamingFakeModel(replies=list(replies))
    monkeypatch.setattr(graph_module, "create_chat_model", lambda: fake)
    return fake


def config(thread_id):
    return {"configurable": {"thread_id": thread_id}}


def user_texts(messages):
    return [m.content for m in messages if isinstance(m, HumanMessage)]


CALCULATOR_CALL = AIMessage(
    "", tool_calls=[{"name": "calculator", "args": {"a": 6, "b": 7, "operation": "multiply"}, "id": "call_1"}]
)


# --- Checkpointing and threads (mocked model) --------------------------------

def test_same_thread_remembers_previous_turns(monkeypatch):
    fake = use_fake(monkeypatch, [AIMessage("Nice to meet you, Abhi."), AIMessage("Your name is Abhi.")])
    graph = graph_module.build_graph(checkpointer=InMemorySaver())

    graph.invoke({"messages": [HumanMessage("My name is Abhi.")]}, config("A"))
    graph.invoke({"messages": [HumanMessage("What is my name?")]}, config("A"))

    # The second model call received the first turn, loaded by the checkpointer.
    assert user_texts(fake.calls[1]) == ["My name is Abhi.", "What is my name?"]
    stored = graph.get_state(config("A")).values["messages"]
    assert [type(m).__name__ for m in stored] == ["HumanMessage", "AIMessage", "HumanMessage", "AIMessage"]


def test_different_threads_are_isolated(monkeypatch):
    fake = use_fake(monkeypatch, [AIMessage("Hello.")])
    graph = graph_module.build_graph(checkpointer=InMemorySaver())

    graph.invoke({"messages": [HumanMessage("My name is Abhi.")]}, config("A"))
    graph.invoke({"messages": [HumanMessage("What is my name?")]}, config("B"))

    assert user_texts(fake.calls[1]) == ["What is my name?"]
    assert len(graph.get_state(config("A")).values["messages"]) == 2
    assert len(graph.get_state(config("B")).values["messages"]) == 2


def test_without_checkpointer_nothing_is_remembered(monkeypatch):
    # Contrast case: same two turns, no checkpointer.
    fake = use_fake(monkeypatch, [AIMessage("Hello.")])
    graph = graph_module.build_graph()

    graph.invoke({"messages": [HumanMessage("My name is Abhi.")]}, config("A"))
    graph.invoke({"messages": [HumanMessage("What is my name?")]}, config("A"))

    assert user_texts(fake.calls[1]) == ["What is my name?"]


def test_checkpointer_requires_thread_id(monkeypatch):
    use_fake(monkeypatch, [AIMessage("Hello.")])
    graph = graph_module.build_graph(checkpointer=InMemorySaver())

    with pytest.raises(ValueError, match="thread_id"):
        graph.invoke({"messages": [HumanMessage("Hi")]})


def test_checkpoint_ids_are_created_automatically(monkeypatch):
    use_fake(monkeypatch, [AIMessage("Hello.")])
    graph = graph_module.build_graph(checkpointer=InMemorySaver())

    graph.invoke({"messages": [HumanMessage("Hi")]}, config("A"))

    saved = graph.get_state(config("A")).config["configurable"]
    assert saved["thread_id"] == "A"
    assert saved["checkpoint_id"]


# --- Streaming (mocked model that genuinely streams) ------------------------

def test_stream_reply_yields_multiple_pieces(monkeypatch):
    use_fake(monkeypatch, [AIMessage("LangGraph streams this reply word by word")])
    graph = graph_module.build_graph(checkpointer=InMemorySaver())

    pieces = list(stream_reply(graph, "Hi", "A"))

    assert len(pieces) > 1
    assert "".join(pieces) == "LangGraph streams this reply word by word"


def test_stream_reply_hides_tool_output(monkeypatch):
    use_fake(monkeypatch, [CALCULATOR_CALL, AIMessage("6 times 7 is 42")])
    graph = graph_module.build_graph(checkpointer=InMemorySaver())

    pieces = list(stream_reply(graph, "What is 6 times 7?", "A"))

    # The tool really ran...
    stored = graph.get_state(config("A")).values["messages"]
    assert [m.content for m in stored if isinstance(m, ToolMessage)] == ["42.0"]
    # ...but the user only sees the final answer, not the raw tool result.
    assert "".join(pieces) == "6 times 7 is 42"
    assert "42.0" not in pieces


def test_streamed_reply_is_saved_to_thread(monkeypatch):
    use_fake(monkeypatch, [AIMessage("Saved after streaming")])
    graph = graph_module.build_graph(checkpointer=InMemorySaver())

    list(stream_reply(graph, "Hi", "A"))

    last = graph.get_state(config("A")).values["messages"][-1]
    assert isinstance(last, AIMessage)
    assert last.content == "Saved after streaming"


def test_stream_reply_uses_thread_memory(monkeypatch):
    fake = use_fake(monkeypatch, [AIMessage("Noted"), AIMessage("Your name is Abhi")])
    graph = graph_module.build_graph(checkpointer=InMemorySaver())

    list(stream_reply(graph, "My name is Abhi.", "A"))
    list(stream_reply(graph, "What is my name?", "A"))

    assert user_texts(fake.calls[1]) == ["My name is Abhi.", "What is my name?"]
