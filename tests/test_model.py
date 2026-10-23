
from unittest.mock import MagicMock

from general_purpose_ai_chatbot.chatbot import model as model_module
from general_purpose_ai_chatbot.chatbot import config as config_module


def test_config_loads_required_settings():
    assert config_module.GROQ_API_KEY
    assert config_module.TAVILY_API_KEY
    assert config_module.GROQ_MODEL


def test_create_chat_model(monkeypatch):
    mock_model = MagicMock()
    mock_chat_groq = MagicMock(return_value=mock_model)

    monkeypatch.setattr(model_module, "ChatGroq", mock_chat_groq)
    monkeypatch.setattr(model_module, "GROQ_API_KEY", "test-api-key")
    monkeypatch.setattr(model_module, "GROQ_MODEL", "test-model")

    result = model_module.create_chat_model()

    mock_chat_groq.assert_called_once_with(
        model="test-model",
        api_key="test-api-key",
        temperature=0.3,
    )
    assert result is mock_model
