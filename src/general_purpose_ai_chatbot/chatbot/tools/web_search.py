from langchain_tavily import TavilySearch

from general_purpose_ai_chatbot.chatbot.config import TAVILY_API_KEY

MAX_RESULTS = 3

def create_web_search_tool() -> TavilySearch:
    return TavilySearch(
        tavily_api_key=TAVILY_API_KEY, 
        max_results=MAX_RESULTS,
        search_depth="basic",
        )
