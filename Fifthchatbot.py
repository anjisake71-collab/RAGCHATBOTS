import os, json
import streamlit as st
from dotenv import load_dotenv

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage

# ---------------- Setup ----------------
load_dotenv()
st.set_page_config(page_title="Chatbot 5", page_icon="💾", layout="centered")
st.title("💾 Chatbot 5 — Persistent Memory (Save/Load)")

key = os.getenv("OPENAI_API_KEY")
if not key:
    st.error("OPENAI_API_KEY not found in .env / environment")
    st.stop()

# File where we persist chat
HISTORY_FILE = "chat_history.json"

# ---------------- Helpers ----------------
def save_history(messages):
    data = []
    for m in messages:
        role = "human" if isinstance(m, HumanMessage) else "ai"
        data.append({"role": role, "content": m.content})
    with open(HISTORY_FILE, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

def load_history():
    if not os.path.exists(HISTORY_FILE):
        return []
    try:
        with open(HISTORY_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)
        msgs = []
        for item in data:
            if item.get("role") == "human":
                msgs.append(HumanMessage(content=item.get("content", "")))
            else:
                msgs.append(AIMessage(content=item.get("content", "")))
        return msgs
    except Exception:
        return []

def clear_history_file():
    if os.path.exists(HISTORY_FILE):
        os.remove(HISTORY_FILE)

# ---------------- Sidebar ----------------
with st.sidebar:
    st.header("⚙️ Settings")
    model = st.selectbox("Model", ["gpt-4o-mini", "gpt-4.1-mini", "gpt-4.1"], index=0)
    temperature = st.slider("Temperature", 0.0, 1.0, 0.2, 0.05)
    clear = st.button("🧹 Clear chat (and file)", use_container_width=True)

# ---------------- LLM ----------------
llm = ChatOpenAI(model=model, api_key=key, temperature=temperature)

# ---------------- Persistent Memory ----------------
if "messages" not in st.session_state:
    st.session_state.messages = load_history()

if clear:
    st.session_state.messages = []
    clear_history_file()
    st.rerun()

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
user_text = st.chat_input("Type your message...")
if user_text:
    st.session_state.messages.append(HumanMessage(content=user_text))
    with st.chat_message("user"):
        st.markdown(user_text)

    ai_text = chain.invoke({"messages": st.session_state.messages}).content

    with st.chat_message("assistant"):
        st.markdown(ai_text)

    st.session_state.messages.append(AIMessage(content=ai_text))

    # Save after every turn
    save_history(st.session_state.messages)
       