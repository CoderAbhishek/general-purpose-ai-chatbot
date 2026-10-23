from general_purpose_ai_chatbot.chatbot.model import create_chat_model
from general_purpose_ai_chatbot.chatbot.state import ChatState
from general_purpose_ai_chatbot.chatbot.tools.web_search import create_web_search_tool
from general_purpose_ai_chatbot.chatbot.tools.calculator import calculator
from general_purpose_ai_chatbot.chatbot.tools.summariser import summarise_text

from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import ToolNode, tools_condition
from langchain_core.messages import SystemMessage

SYSTEM_PROMPT = (
    "You are a helpful general-purpose assistant. "
    "Use the calculator for arithmetic, summarise_text when asked to summarise text, "
    "and tavily_search for current or recent information. "
    "When you use search results, cite the source URLs."
)

def build_graph():
    tools = [calculator, summarise_text, create_web_search_tool()]
    model = create_chat_model().bind_tools(tools)

    def call_model(state: ChatState):
        response = model.invoke([SystemMessage(content=SYSTEM_PROMPT)] + state["messages"])
        return {"messages": [response]}

    builder = StateGraph(ChatState)
    builder.add_node("llm", call_model)
    builder.add_node("tools", ToolNode(tools, handle_tool_errors=True))
    builder.add_edge(START, "llm")
    builder.add_conditional_edges("llm", tools_condition)
    builder.add_edge("tools", "llm")
    return builder.compile()