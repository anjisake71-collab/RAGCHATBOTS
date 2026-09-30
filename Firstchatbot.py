import streamlit as st
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate

# ✅ DEMO ONLY (do NOT do this in real projects)
OPENAI_API_KEY = "PASTE_YOUR_OPENAI_KEY_HERE"

st.set_page_config(page_title="Chatbot v1 (Key in Code)", page_icon="🤖")
st.title("🤖 Chatbot v1 — API key inside code (Demo)")

# Create model
llm = ChatOpenAI(
    model="gpt-4o-mini",       # change if you want
    api_key=OPENAI_API_KEY,
    temperature=0.2
)

# Prompt
prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant."),
    ("human", "{user_input}")
])

user_input = st.text_input("Type your message:")

if st.button("Send") and user_input.strip():
    chain = prompt | llm
    result = chain.invoke({"user_input": user_input})
    st.write("**Assistant:**")
    st.write(result.content)
       