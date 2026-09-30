import os
import json
import requests
import streamlit as st
from dotenv import load_dotenv

from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage, ToolMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

load_dotenv()

st.set_page_config(page_title="Weather Chatbot", page_icon="🌦️")
st.title("🌦️ Weather Chatbot (Tool Calling)")

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENWEATHER_API_KEY = os.getenv("OPENWEATHER_API_KEY")

if not OPENAI_API_KEY:
    st.error("OPENAI_API_KEY not found in .env / environment")
    st.stop()

if not OPENWEATHER_API_KEY:
    st.error("OPENWEATHER_API_KEY not found in .env / environment")
    st.stop()


@tool
def get_weather(city: str) -> dict:
    """Get current weather for a city using OpenWeather API."""
    url = (
        "http://api.openweathermap.org/data/2.5/weather"
        f"?q={city}&appid={OPENWEATHER_API_KEY}&units=metric"
    )
    r = requests.get(url, timeout=15)

    if r.status_code == 200:
        d = r.json()
        return {
            "city": d["name"],
            "country": d["sys"]["country"],
            "temp_c": d["main"]["temp"],
            "feels_like_c": d["main"]["feels_like"],
            "description": d["weather"][0]["description"],
        }
    if r.status_code == 404:
        return {"error": f"City '{city}' not found"}
    if r.status_code == 401:
        return {"error": "Invalid OpenWeather API key"}
    return {"error": f"OpenWeather API error: {r.status_code}"}


llm = ChatOpenAI(
    model="gpt-4o-mini",
    api_key=OPENAI_API_KEY,
    temperature=0.2,
).bind_tools([get_weather])

prompt = ChatPromptTemplate.from_messages([
    ("system",
     "You are a helpful assistant. If the user asks about weather in a city, "
     "call the get_weather tool. If the city is missing, ask which city."),
    MessagesPlaceholder("messages"),
])

chain = prompt | llm

if "messages" not in st.session_state:
    st.session_state.messages = []

# Show history (only user + assistant text)
for m in st.session_state.messages:
    if isinstance(m, ToolMessage):
        continue
    role = "user" if m.type == "human" else "assistant"
    with st.chat_message(role):
        st.markdown(m.content)

user_text = st.chat_input("Ask: What's the weather in Hyderabad?")
if user_text:
    st.session_state.messages.append(HumanMessage(content=user_text))
    with st.chat_message("user"):
        st.markdown(user_text)

    ai_msg = chain.invoke({"messages": st.session_state.messages})

    # If model asked to call tool(s), execute and send tool results back
    if getattr(ai_msg, "tool_calls", None):
        st.session_state.messages.append(ai_msg)

        for call in ai_msg.tool_calls:
            result = get_weather.invoke(call.get("args", {}))
            st.session_state.messages.append(
                ToolMessage(content=json.dumps(result), tool_call_id=call["id"])
            )

        final_msg = chain.invoke({"messages": st.session_state.messages})
        st.session_state.messages.append(final_msg)

        with st.chat_message("assistant"):
            st.markdown(final_msg.content)

    else:
        st.session_state.messages.append(ai_msg)
        with st.chat_message("assistant"):
            st.markdown(ai_msg.content)
    