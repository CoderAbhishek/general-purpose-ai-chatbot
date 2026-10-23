from pathlib import Path

import pytest
import streamlit as st
from langchain_core.messages import AIMessage
from streamlit.testing.v1 import AppTest

from fakes import StreamingFakeModel
from general_purpose_ai_chatbot.chatbot import graph as graph_module

APP_PATH = str(Path(__file__).resolve().parent.parent / "app.py")

CALCULATOR_CALL = AIMessage(
    "", tool_calls=[{"name": "calculator", "args": {"a": 1234, "b": 5678, "operation": "multiply"}, "id": "c1"}]
)


class FailingModel(StreamingFakeModel):
    """Fails while answering, like a network error in the middle of a turn."""

    def _next_reply(self, messages):
        raise RuntimeError("Groq unreachable")


@pytest.fixture
def start_app(monkeypatch):
    """Run app.py headlessly with a fake model. The graph cache is cleared so each test starts fresh."""

    def start(model):
        monkeypatch.setattr(graph_module, "create_chat_model", lambda: model)
        st.cache_resource.clear()
        return AppTest.from_file(APP_PATH, default_timeout=30).run()

    yield start
    st.cache_resource.clear()


def messages(app):
    return [(m.name, m.markdown[0].value) for m in app.chat_message]


def captions_in_chat(app):
    return [c.value for m in app.chat_message for c in m.caption]


# --- First load -----------------------------------------------------------

def test_first_load_shows_welcome_screen(start_app):
    app = start_app(StreamingFakeModel(replies=[AIMessage("unused")]))

    assert not app.exception
    assert app.title[0].value == "🤖 General-Purpose AI Chatbot"
    assert messages(app) == [("assistant", "Hi! I'm your AI assistant. Ask me anything, or try one of these:")]
    assert [b.label for b in app.main.button] == ["🧮 Do a calculation", "🌐 Search the web", "📝 Summarise text"]
    assert [b.label for b in app.sidebar.button] == ["➕ New conversation"]
    assert app.sidebar.caption[0].value == f"Conversation ID: `{app.session_state.thread_id[:8]}`"


# --- Chatting -------------------------------------------------------------

def test_typed_message_gets_streamed_reply(start_app):
    app = start_app(StreamingFakeModel(replies=[AIMessage("Hello Abhi, how can I help?")]))

    app.chat_input[0].set_value("Hi").run()

    assert not app.exception
    assert messages(app) == [("user", "Hi"), ("assistant", "Hello Abhi, how can I help?")]
    assert len(app.main.button) == 0  # welcome buttons gone
    assert captions_in_chat(app) == []  # no tools used, no caption


def test_example_button_sends_prompt_and_shows_tools_used(start_app):
    app = start_app(StreamingFakeModel(replies=[CALCULATOR_CALL, AIMessage("1234 times 5678 is 7006652.")]))

    app.main.button[0].click().run()

    assert not app.exception
    assert messages(app) == [
        ("user", "What is 1,234 multiplied by 5,678?"),
        ("assistant", "1234 times 5678 is 7006652."),
    ]
    assert captions_in_chat(app) == ["🔧 Used: Calculator"]


def test_conversation_continues_across_reruns(start_app):
    fake = StreamingFakeModel(replies=[AIMessage("Nice to meet you."), AIMessage("Your name is Abhi.")])
    app = start_app(fake)
    thread_id = app.session_state.thread_id

    app.chat_input[0].set_value("My name is Abhi.").run()
    app.chat_input[0].set_value("What is my name?").run()

    assert app.session_state.thread_id == thread_id
    assert len(messages(app)) == 4
    sent_to_model = [m.content for m in fake.calls[-1]]
    assert "My name is Abhi." in sent_to_model


def test_new_conversation_starts_empty_thread(start_app):
    app = start_app(StreamingFakeModel(replies=[AIMessage("Hello!")]))
    app.chat_input[0].set_value("Hi").run()
    old_thread = app.session_state.thread_id

    app.sidebar.button[0].click().run()

    assert app.session_state.thread_id != old_thread
    assert messages(app)[0][1].startswith("Hi! I'm your AI assistant")  # welcome screen back
    assert ("user", "Hi") not in messages(app)


# --- Failure --------------------------------------------------------------

def test_error_during_reply_shows_message_instead_of_crashing(start_app):
    app = start_app(FailingModel(replies=[AIMessage("unused")]))

    app.chat_input[0].set_value("Hi").run()

    assert not app.exception
    assert "Groq unreachable" in app.error[0].value
    assert not any("Thinking" in md.value for m in app.chat_message for md in m.markdown)
