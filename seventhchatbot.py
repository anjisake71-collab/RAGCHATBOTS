import os
import streamlit as st
from dotenv import load_dotenv

from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage
from langchain_google_genai import ChatGoogleGenerativeAI

# ---------------- Setup ----------------
load_dotenv()
st.set_page_config(page_title="Chatbot 7 — Gemini", page_icon="✨", layout="centered")
st.title("✨ Chatbot 7 — Gemini (LangChain + Modern Memory)")

google_key = os.getenv("GOOGLE_API_KEY")
if not google_key:
    st.error("GOOGLE_API_KEY not found. Add it in .env or set it in your system environment.")
    st.stop()

# ---------------- Sidebar ----------------
with st.sidebar:
    st.header("⚙️ Settings")

    # Default to a currently documented model name.
    # If this ever breaks, run the list_models script and paste the exact model name.
    model = st.text_input("Gemini model name", value="gemini-2.5-flash")

    temperature = st.slider("Temperature", 0.0, 1.0, 0.2, 0.05)
    window_k = st.number_input("Keep last K turns (0 = keep all)", 0, 50, 10, 1)
    clear = st.button("🧹 Clear chat", use_container_width=True)

# ---------------- LLM ----------------
llm = ChatGoogleGenerativeAI(
    model=model,
    temperature=temperature,
    google_api_key=google_key,
)     

# ---------------- Memory ----------------
if "messages" not in st.session_state:
    st.session_state.messages = []  # list[HumanMessage | AIMessage]

if clear:
    st.session_state.messages = []
    st.rerun()

def apply_window(msgs, k: int):
    if k <= 0:
        return msgs
    return msgs[-2 * k:]  # each turn = (human + ai)

# ---------------- Prompt / Chain ----------------
prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant."),
    MessagesPlaceholder("messages"),
])
chain = prompt | llm

# ---------------- Render history ----------------
for m in st.session_state.messages:
    role = "user" if isinstance(m, HumanMessage) else "assistant"
    with st.chat_message(role):
        st.markdown(m.content)

# ---------------- Chat input ----------------
user_text = st.chat_input("Ask me anything...")
if user_text:
    st.session_state.messages.append(HumanMessage(content=user_text))
    with st.chat_message("user"):
        st.markdown(user_text)

    history = apply_window(st.session_state.messages, int(window_k))
    ai_msg = chain.invoke({"messages": history})

    ai_text = ai_msg.content if hasattr(ai_msg, "content") else str(ai_msg)

    with st.chat_message("assistant"):
        st.markdown(ai_text)

    st.session_state.messages.append(AIMessage(content=ai_text))
       