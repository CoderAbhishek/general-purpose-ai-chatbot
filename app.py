from uuid import uuid4

import streamlit as st
from langgraph.checkpoint.memory import InMemorySaver

from general_purpose_ai_chatbot.chatbot.graph import build_graph, get_chat_history, stream_reply

AVATARS = {"user": "🧑", "assistant": "🤖"}
TOOL_LABELS = {"calculator": "Calculator", "summarise_text": "Summariser", "tavily_search": "Web search"}
EXAMPLES = [
    ("🧮 Do a calculation", "What is 1,234 multiplied by 5,678?"),
    ("🌐 Search the web", "What are today's top technology news headlines?"),
    (
        "📝 Summarise text",
        "Summarise this in two sentences: Streamlit is an open-source Python framework for building "
        "data apps. Developers write ordinary Python scripts, and Streamlit turns each call such as "
        "st.title or st.chat_message into an interactive web component. Every user interaction reruns "
        "the script from top to bottom, so app state that must survive is kept in session state or a cache.",
    ),
]


@st.cache_resource
def get_graph():
    return build_graph(checkpointer=InMemorySaver())


def show_tools_used(tools):
    if tools:
        st.caption("🔧 Used: " + ", ".join(TOOL_LABELS.get(name, name) for name in tools))


def with_thinking_indicator(pieces, placeholder):
    """Show 'Thinking...' until the first piece of the reply arrives."""
    placeholder.markdown("_Thinking..._")
    for piece in pieces:
        placeholder.empty()
        yield piece


st.set_page_config(page_title="General-Purpose AI Chatbot", page_icon="🤖")

if "thread_id" not in st.session_state:
    st.session_state.thread_id = str(uuid4())

with st.sidebar:
    st.header("🤖 AI Chatbot")
    if st.button("➕ New conversation", width="stretch"):
        st.session_state.thread_id = str(uuid4())
    st.caption(f"Conversation ID: `{st.session_state.thread_id[:8]}`")
    st.divider()
    st.subheader("What I can do")
    st.markdown(
        "- 🧮 **Calculator**: exact arithmetic\n"
        "- 📝 **Summariser**: condense long text\n"
        "- 🌐 **Web search**: current information with sources"
    )
    st.divider()
    st.caption("Memory lasts for this session only. Powered by Groq, LangGraph and Tavily.")

st.title("🤖 General-Purpose AI Chatbot")
st.caption("Ask questions, do maths, summarise text and search the web.")

graph = get_graph()
history = get_chat_history(graph, st.session_state.thread_id)

example_prompt = None
if not history:
    with st.chat_message("assistant", avatar=AVATARS["assistant"]):
        st.markdown("Hi! I'm your AI assistant. Ask me anything, or try one of these:")
    for column, (label, text) in zip(st.columns(len(EXAMPLES)), EXAMPLES):
        if column.button(label, width="stretch"):
            example_prompt = text

for item in history:
    with st.chat_message(item["role"], avatar=AVATARS[item["role"]]):
        st.markdown(item["text"])
        show_tools_used(item["tools"])

prompt = st.chat_input("Ask me anything") or example_prompt
if prompt:
    with st.chat_message("user", avatar=AVATARS["user"]):
        st.markdown(prompt)
    with st.chat_message("assistant", avatar=AVATARS["assistant"]):
        placeholder = st.empty()
        try:
            st.write_stream(with_thinking_indicator(stream_reply(graph, prompt, st.session_state.thread_id), placeholder))
        except Exception as error:
            placeholder.empty()
            st.error(f"Something went wrong: {error}")
        else:
            # Redraw from the saved history: hides the welcome screen and adds the "Used" caption.
            st.rerun()
