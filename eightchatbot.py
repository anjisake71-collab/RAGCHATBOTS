import os
import streamlit as st
from dotenv import load_dotenv

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage

# ---------------- Setup ----------------
load_dotenv()
st.set_page_config(page_title="Chatbot 8 — DeepSeek", page_icon="🧠", layout="centered")
st.title("🧠 Chatbot 8 — DeepSeek (LangChain + Modern Memory)")

DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY")
if not DEEPSEEK_API_KEY:
    st.error("DEEPSEEK_API_KEY not found. Add it in .env or set it in your system environment.")
    st.stop()

# DeepSeek is OpenAI-compatible. Use base_url with /v1 for compatibility.
DEEPSEEK_BASE_URL = "https://api.deepseek.com/v1"

# ---------------- Sidebar UI ----------------
with st.sidebar:
    st.header("⚙️ Settings")

    model = st.selectbox(
        "Model",
        ["deepseek-chat", "deepseek-reasoner"],  # from DeepSeek docs
        index=0
    )

    temperature = st.slider("Temperature", 0.0, 1.0, 0.2, 0.05)

    window_k = st.number_input(
        "Keep last K turns (0 = keep all)",
        min_value=0, max_value=50, value=10, step=1
    )

    show_debug = st.checkbox("Show debug (last request context)", value=False)

    clear = st.button("🧹 Clear chat", use_container_width=True)

# ---------------- LLM ----------------
llm = ChatOpenAI(
    model=model,
    api_key=DEEPSEEK_API_KEY,
    base_url=DEEPSEEK_BASE_URL,
    temperature=temperature,
)

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

# ---------------- Render history ----------------
for m in st.session_state.messages:
    role = "user" if isinstance(m, HumanMessage) else "assistant"
    with st.chat_message(role):
        st.markdown(m.content)

# ---------------- Chat input ----------------
user_text = st.chat_input("Ask me anything...")
if user_text:
    # Store + show user msg
    st.session_state.messages.append(HumanMessage(content=user_text))
    with st.chat_message("user"):
        st.markdown(user_text)

    # Windowed history
    history = apply_window(st.session_state.messages, int(window_k))

    if show_debug:
        st.sidebar.write("### Debug: Messages sent (windowed)")
        st.sidebar.write([{"type": m.type, "content": m.content[:200]} for m in history])

    # Invoke
    ai_msg = chain.invoke({"messages": history})
    ai_text = ai_msg.content if hasattr(ai_msg, "content") else str(ai_msg)

    # Show + store assistant msg
    with st.chat_message("assistant"):
        st.markdown(ai_text)
    st.session_state.messages.append(AIMessage(content=ai_text))
   