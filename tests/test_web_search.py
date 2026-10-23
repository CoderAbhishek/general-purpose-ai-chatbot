import pytest
from langchain_tavily import TavilySearch
from langchain_tavily._utilities import TavilySearchAPIWrapper

from general_purpose_ai_chatbot.chatbot.tools import web_search as web_search_module
from general_purpose_ai_chatbot.chatbot.tools.web_search import MAX_RESULTS, create_web_search_tool


# --- Unit: configuration and schema (no network) ---------------------------

def test_factory_returns_tavily_tool():
    tool = create_web_search_tool()
    assert isinstance(tool, TavilySearch)
    assert tool.name == "tavily_search"


def test_cost_settings_are_locked():
    tool = create_web_search_tool()
    assert MAX_RESULTS == 3
    assert tool.max_results == MAX_RESULTS
    assert tool.search_depth == "basic"


def test_api_key_comes_from_config(monkeypatch):
    # The environment and config normally hold the same key, so give config a
    # different value: this proves the key is passed explicitly, not read from the environment.
    monkeypatch.setattr(web_search_module, "TAVILY_API_KEY", "key-from-config")
    tool = create_web_search_tool()
    assert tool.api_wrapper.tavily_api_key.get_secret_value() == "key-from-config"


def test_schema_exposes_query_but_not_cost_settings():
    tool = create_web_search_tool()
    assert "query" in tool.args
    assert "max_results" not in tool.args


def test_errors_are_returned_to_the_model():
    assert create_web_search_tool().handle_tool_error is True


# --- Mocked: replace only the HTTP call to Tavily --------------------------

@pytest.fixture
def fake_tavily(monkeypatch):
    """Replace the Tavily HTTP call; record what was sent and control what comes back."""
    state = {"calls": [], "response": {"results": [{"title": "T", "url": "https://example.com", "content": "C", "score": 0.9}]}}

    def fake_raw_results(self, **kwargs):
        state["calls"].append(kwargs)
        if isinstance(state["response"], Exception):
            raise state["response"]
        return state["response"]

    monkeypatch.setattr(TavilySearchAPIWrapper, "raw_results", fake_raw_results)
    return state


def test_returns_results_with_urls(fake_tavily):
    result = create_web_search_tool().invoke({"query": "langgraph"})
    assert result["results"][0]["url"] == "https://example.com"
    assert fake_tavily["calls"][0]["query"] == "langgraph"


def test_model_cannot_override_locked_settings(fake_tavily):
    create_web_search_tool().invoke({"query": "langgraph", "search_depth": "advanced"})
    sent = fake_tavily["calls"][0]
    assert sent["search_depth"] == "basic"
    assert sent["max_results"] == MAX_RESULTS


def test_no_results_becomes_message_for_model(fake_tavily):
    fake_tavily["response"] = {"results": []}
    result = create_web_search_tool().invoke({"query": "zzqqxx"})
    assert isinstance(result, str)
    assert "No search results found" in result


def test_network_failure_is_returned_not_raised(fake_tavily):
    fake_tavily["response"] = ConnectionError("network down")
    result = create_web_search_tool().invoke({"query": "langgraph"})
    assert "error" in result
