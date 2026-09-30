import os
import streamlit as st
from dotenv import load_dotenv
from openai import OpenAI

# ---------------- Setup ----------------
load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

st.set_page_config(page_title="Basic AI Chatbot (No Tools)", page_icon="🤖", layout="centered")
st.title("🤖 Basic AI Chatbot (No Tools / No Real-Time)")
st.caption("Uses only the model's knowledge. No web, no weather API, no live data.")

# ---------------- Memory (Session State) ----------------
# This list stays across Streamlit reruns (until refresh/clear)
if "messages" not in st.session_state:
    st.session_state.messages = []  # list[{"role": "...", "content": "..."}]

# ---------------- Render chat history ----------------
for m in st.session_state.messages:
    with st.chat_message(m["role"]):
        st.markdown(m["content"])

# ---------------- Sidebar (Student Notes) ----------------
with st.sidebar:
    st.header("📚 Student Notes (Limitations Demo)")
    st.info(
        "This bot has **NO tools**. It cannot access live weather/news/stock prices.\n\n"
        "Try asking:\n"
        "• What's the weather in London today?\n"
        "• What's the current temperature in Paris?\n"
        "• Is it raining in Tokyo right now?\n"
    )

    if os.getenv("OPENAI_API_KEY"):
        st.success("✅ OpenAI API key found")
    else:
        st.error("❌ OPENAI_API_KEY not found")

    if st.button("🧹 Clear chat", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

# ---------------- Chat input ----------------
user_text = st.chat_input("Ask me anything...")
if user_text:
    # 1) store + show user msg
    st.session_state.messages.append({"role": "user", "content": user_text})
    with st.chat_message("user"):
        st.markdown(user_text)

    # 2) system message for limitations
    system_msg = {
        "role": "system",
        "content": (
            "You are a helpful assistant. You do NOT have access to real-time information "
            "like current weather, live news, today's stock prices, or the internet. "
            "If asked for real-time info, clearly say you can't access it and offer general guidance."
        )
    }

    # 3) call OpenAI (modern Responses API)
    try:
        resp = client.responses.create(
            model="gpt-4o-mini",
            input=[system_msg, *st.session_state.messages],
        )

        ai_text = resp.output_text

        # 4) show + store assistant msg
        with st.chat_message("assistant"):
            st.markdown(ai_text)

        st.session_state.messages.append({"role": "assistant", "content": ai_text})

    except Exception as e:
        st.error(f"Error: {e}")    
    