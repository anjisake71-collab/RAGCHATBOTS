# 🤖 AI Chatbots, Tool Calling & RAG Applications

A progressive collection of **AI chatbot and Retrieval-Augmented Generation (RAG) applications** built with **Python, Streamlit, OpenAI, LangChain, Gemini, DeepSeek, FAISS, Chroma, and DuckDB**.

This repository demonstrates the step-by-step evolution of AI applications — from a simple chatbot to **memory-enabled assistants, tool-calling agents, document-based RAG systems, website Q&A, CSV analysis, and natural-language SQL querying**.

---

## 🚀 Features

### 🤖 Chatbot Evolution
- Basic LLM chatbot
- Secure API-key configuration using `.env`
- Conversational memory
- Configurable conversation history
- Improved Streamlit UI
- Persistent chat history

### 🧠 Multiple LLM Providers
- OpenAI
- Google Gemini
- DeepSeek

### 🛠️ Tool Calling
- 🌦️ Real-time weather assistant
- 💱 Live currency conversion
- External API integration with LLM tool calling

### 📚 RAG Applications
- 📄 Text document Q&A
- 📑 Multiple PDF Q&A
- 🗂️ PDF RAG with Chroma
- 📊 CSV data Q&A
- 🌐 Website content Q&A
- 🔎 Semantic search using vector databases

### 🧮 Structured Data Assistant
- Natural-language questions over business data
- LLM-generated SQL queries
- DuckDB query execution
- CSV data loading
- Query result tables
- Summary statistics

---

## 🛠️ Tech Stack

| Technology | Purpose |
|---|---|
| 🐍 Python | Application development |
| 🎈 Streamlit | Web interface |
| 🦜 LangChain | LLM & RAG orchestration |
| 🤖 OpenAI | Language models & embeddings |
| ✨ Google Gemini | Alternative LLM provider |
| 🧠 DeepSeek | Alternative LLM provider |
| 🔎 FAISS | Vector similarity search |
| 🗃️ Chroma | Vector database |
| 🐼 Pandas | Data processing |
| 🦆 DuckDB | SQL analytics |
| 📄 PyPDF2 | PDF processing |
| 🌐 WebBaseLoader | Website data loading |

---

## 📦 Project Structure

```text
RAGCHATBOTS/
│
├── Firstchatbot.py
├── Second_chatbot.py
├── Thirdchatbot.py
├── Fourth_chatbot.py
├── Fifth_chatbot.py
│
├── sixthchat2.py
├── sixthchatbot3.py
│
├── seventhchatbot.py
├── Eigth_chatbot.py
│
├── Ninth_chatbot.py
├── tenthchatbot.py
├── Eleventh_chatbot.py
├── twelfthchatbot.py
├── thirteenchatbot.py
│
├── 14.Fourteenth_chatbot/
│   ├── app.py
│   ├── data/
│   ├── llm_interface/
│   ├── prompts/
│   ├── query_engine/
│   ├── utils/
│   ├── requirements.txt
│   └── README.md
│
├── requirements.txt
└── README.md

📌 Sample Use Cases
💬 General Chat
User: Explain Retrieval-Augmented Generation.

Assistant: RAG combines information retrieval with a language
model to generate answers using relevant external knowledge.
🌦️ Weather Tool
User: What's the weather in London?

Assistant: Uses the weather API tool to retrieve current data
and provide the result.
📄 Document RAG
User: What does the uploaded document say about revenue?

Assistant: Searches relevant document chunks and answers
using the retrieved context.
🧮 Business Data Q&A
User: What was the total revenue for India?

Assistant:
1. Generates SQL from the question
2. Executes SQL using DuckDB
3. Displays the result
4. Provides a summary
⚙️ Installation
git clone https://github.com/anjsake71-collab/RAGCHATBOTS.git
cd RAGCHATBOTS

python -m venv venv
Windows
venv\Scripts\activate
Install dependencies
pip install -r requirements.txt
🔑 Environment Variables

Create a .env file:

OPENAI_API_KEY=your_openai_api_key
GOOGLE_API_KEY=your_google_api_key
DEEPSEEK_API_KEY=your_deepseek_api_key
EXCHANGE_API_KEY=your_exchange_api_key

🔮 Future Enhancements
🔐 Improved authentication and access control
📊 Advanced analytics dashboards
💾 Production-ready vector database storage
⚡ Streaming responses
🧠 Agentic RAG workflows
🔗 More external tools and APIs
☁️ Cloud deployment
🗂️ Better document management
📈 Evaluation and monitoring
🛡️ Security

This repository is intended for learning and demonstration purposes.

Store API keys in environment variables.
Never commit secrets to GitHub.
Do not expose production credentials in source code.
Use .gitignore to protect local configuration and runtime files.
📚 License

This project is provided for educational and learning purposes.

🙌 Acknowledgements

Built while exploring modern Generative AI, LLM applications, LangChain, RAG, vector databases, tool calling, and structured-data querying.

⭐ Support

If you find this repository useful, consider giving it a ⭐ on GitHub.


**One important thing before pushing:** your 14th project contains an API key directly in the Python source. **Remove/rotate that key before committing or pushing to GitHub**; `.gitignore` will not remove a secret that is already tracked.
      
