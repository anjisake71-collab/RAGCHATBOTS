import os, json, requests
import streamlit as st
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

st.set_page_config(page_title="Currency Chatbot", page_icon="💱", layout="centered")
st.title("💱 Currency Chatbot (Tool Calling)")

# ---------- Memory ----------
if "messages" not in st.session_state:
    st.session_state.messages = []

# ---------- Currency function ----------
def convert_currency(amount: float, from_currency: str, to_currency: str):
    api_key = os.getenv("EXCHANGE_API_KEY")
    if not api_key:
        return {"error": "EXCHANGE_API_KEY missing in .env"}

    url = f"https://v6.exchangerate-api.com/v6/{api_key}/pair/{from_currency}/{to_currency}/{amount}"
    
    try:
        r = requests.get(url, timeout=15)
        data = r.json()

        st.sidebar.write("API Response:", data)  # Debugging line

        if r.status_code != 200 or data.get("result") != "success":
            return {"error": f"Conversion failed: {data}", "raw": data}

        return {
            "amount": amount,
            "from_currency": from_currency.upper(),
            "to_currency": to_currency.upper(),
            "rate": data.get("conversion_rate"),
            "converted_amount": data.get("conversion_result"),
        }
    except Exception as e:
        return {"error": f"Request failed: {str(e)}"}

# ---------- Tool schema ----------
tools = [{
    "type": "function",
    "function": {  # Correct tool definition
        "name": "convert_currency",
        "description": "Convert money between currencies using live exchange rates.",
        "parameters": {
            "type": "object",
            "properties": {
                "amount": {"type": "number", "description": "Amount to convert"},
                "from_currency": {"type": "string", "description": "Source currency code"},
                "to_currency": {"type": "string", "description": "Target currency code"}
            },
            "required": ["amount", "from_currency", "to_currency"],
        },
    }
}]

# ---------- Sidebar ----------
with st.sidebar:
    st.write("### API Configuration")
    st.write(f"OpenAI Key: {'✅ Present' if os.getenv('OPENAI_API_KEY') else '❌ Missing'}")
    st.write(f"Exchange API Key: {'✅ Present' if os.getenv('EXCHANGE_API_KEY') else '❌ Missing'}")
    
    if st.button("🧹 Clear chat", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

# ---------- Render history ----------
for m in st.session_state.messages:
    with st.chat_message(m["role"]):
        st.markdown(m["content"])

# ---------- Chat ----------
user_text = st.chat_input("e.g., Convert 100 USD to INR")
if user_text:
    # Append user message
    st.session_state.messages.append({"role": "user", "content": user_text})
    with st.chat_message("user"):
        st.markdown(user_text)

    try:
        # Prepare messages for API call
        messages = [
            {"role": "system", "content": "You are a helpful currency conversion assistant. Use the convert_currency tool for live conversions."}
        ]
        messages.extend([{"role": m["role"], "content": m["content"]} for m in st.session_state.messages])

        # First API call to check for tool usage
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=messages,
            tools=tools,
            tool_choice="auto"
        )

        # Get the response message
        response_message = response.choices[0].message

        # Check if tools were called
        if response_message.tool_calls:
            # Process the first tool call
            tool_call = response_message.tool_calls[0]
            
            # Parse arguments safely
            try:
                args = json.loads(tool_call.function.arguments)
                
                # Perform currency conversion
                conversion_result = convert_currency(
                    args["amount"], 
                    args["from_currency"], 
                    args["to_currency"]
                )

                # Prepare messages for second API call
                # KEY CHANGE: Include the original tool call in messages
                messages.append({
                    "role": "assistant",
                    "tool_calls": response_message.tool_calls,
                    "content": None
                })
                messages.append({
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "content": json.dumps(conversion_result)
                })

                # Second API call to get final response
                final_response = client.chat.completions.create(
                    model="gpt-4o-mini",
                    messages=messages,
                    tools=tools
                )

                # Get the final AI response
                ai_text = final_response.choices[0].message.content

            except (json.JSONDecodeError, KeyError) as e:
                ai_text = f"Error parsing conversion arguments: {str(e)}"
        else:
            # No tool was called, use the original response
            ai_text = response_message.content

        # Display and store the response
        with st.chat_message("assistant"):
            st.markdown(ai_text)
        
        st.session_state.messages.append({"role": "assistant", "content": ai_text})

    except Exception as e:
        st.error(f"An error occurred: {str(e)}")
        st.sidebar.error(f"Full Error: {str(e)}")       