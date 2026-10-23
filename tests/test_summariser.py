from unittest.mock import MagicMock

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from general_purpose_ai_chatbot.chatbot.tools import summariser as summariser_module
from general_purpose_ai_chatbot.chatbot.tools.summariser import summarise_text


def use_fake_model(monkeypatch, reply="short summary"):
    fake_model = MagicMock()
    fake_model.invoke.return_value = AIMessage(content=reply)
    fake_factory = MagicMock(return_value=fake_model)
    monkeypatch.setattr(summariser_module, "create_chat_model", fake_factory)
    return fake_model, fake_factory


def test_returns_model_summary(monkeypatch):
    use_fake_model(monkeypatch)
    assert summarise_text.invoke({"text": "Some long text."}) == "short summary"


def test_sends_instruction_and_user_text(monkeypatch):
    fake_model, _ = use_fake_model(monkeypatch)
    summarise_text.invoke({"text": "Some long text."})

    messages = fake_model.invoke.call_args.args[0]
    assert isinstance(messages[0], SystemMessage)
    assert messages[0].content == summariser_module.SUMMARY_INSTRUCTION
    assert isinstance(messages[1], HumanMessage)
    assert messages[1].content == "Some long text."


def test_blank_text_returns_error_without_calling_model(monkeypatch):
    _, fake_factory = use_fake_model(monkeypatch)
    result = summarise_text.invoke({"text": "   "})

    assert result.startswith("Error:")
    fake_factory.assert_not_called()


def test_tool_schema():
    assert summarise_text.name == "summarise_text"
    assert list(summarise_text.args) == ["text"]
