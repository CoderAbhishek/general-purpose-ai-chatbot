from langchain_groq import ChatGroq

from general_purpose_ai_chatbot.chatbot.config import GROQ_MODEL, GROQ_API_KEY


def create_chat_model() -> ChatGroq:
    return ChatGroq(
        model=GROQ_MODEL,
        api_key=GROQ_API_KEY,
        temperature=0.3,
    )