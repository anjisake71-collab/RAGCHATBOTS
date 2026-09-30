import os
import streamlit as st
from dotenv import load_dotenv

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate

load_dotenv()  # loads variables from .env into environment

st.set_page_config(page_title="Chatbot v2 (Env Key)", page_icon="🤖")
st.title("🤖 Chatbot v2")

api_key = os.getenv("OPENAI_API_KEY")

if not api_key:
    st.error("OPENAI_API_KEY not found. Add it in .env or set it in your system environment.")
    st.stop()

llm = ChatOpenAI(
    model="gpt-4o-mini",
    api_key=api_key,
    temperature=0.2
)

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
   
