#Same chatbot as 3 but with better UI
import os
import streamlit as st
from dotenv import load_dotenv

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage

# ---------------- Setup ----------------
load_dotenv()
st.set_page_config(page_title="Chatbot 4", page_icon="🧠", layout="centered")
st.title("🧠 Chatbot 4 — Modern Memory (Better UI)")

key = os.getenv("OPENAI_API_KEY")  
if not key:
    st.error("OPENAI_API_KEY not found in .env / environment")
    st.stop()

# ---------------- Sidebar UI ----------------
with st.sidebar:
    st.header("⚙️ Settings")

    model = st.selectbox(
        "Model",
        ["gpt-4o-mini", "gpt-4.1-mini", "gpt-4.1","gpt-5.6-luna"],
        index=0
    )

    temperature = st.slider("Temperature", 0.0, 1.0, 0.2, 0.05)

    window_k = st.number_input(
        "Keep last K turns (0 = keep all)",
        min_value=0, max_value=50, value=10, step=1    
    )

    clear = st.button("🧹 Clear chat", use_container_width=True)

# ---------------- LLM ----------------
llm = ChatOpenAI(model=model, api_key=key, temperature=temperature)

# ---------------- Modern Memory (Explicit Messages) ----------------
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

# ---------------- Render chat history ----------------
for m in st.session_state.messages:
    role = "user" if isinstance(m, HumanMessage) else "assistant"
    with st.chat_message(role):
        st.markdown(m.content)              

# ---------------- Chat input ----------------
user_text = st.chat_input("Type your message...")
if user_text:
    # Show + store user message
    st.session_state.messages.append(HumanMessage(content=user_text))
    with st.chat_message("user"):
        st.markdown(user_text)

    # Invoke with windowed history
    history = apply_window(st.session_state.messages, int(window_k))
    ai_text = chain.invoke({"messages": history}).content

    # Show + store assistant message
    with st.chat_message("assistant"):
        st.markdown(ai_text)
    st.session_state.messages.append(AIMessage(content=ai_text))
   