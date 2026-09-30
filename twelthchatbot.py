import os
import streamlit as st
import pandas as pd
from dotenv import load_dotenv

from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import CharacterTextSplitter
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage


# -------------------------
# CSV -> TEXT
# -------------------------
def csv_to_text(df: pd.DataFrame) -> str:
    lines = []
    for _, row in df.iterrows():
        row_text = ", ".join([f"{col}: {row[col]}" for col in df.columns])
        lines.append(row_text)
    return "\n".join(lines)


# -------------------------
# TEXT -> CHUNKS
# -------------------------
def get_text_chunks(text: str):
    splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=500,
        chunk_overlap=20,
        length_function=len
    )
    return splitter.split_text(text)


# -------------------------
# CHUNKS -> VECTORSTORE
# -------------------------
@st.cache_resource
def build_vectorstore(text_chunks):
    embeddings = OpenAIEmbeddings()
    vectorstore = FAISS.from_texts(text_chunks, embedding=embeddings)
    return vectorstore


def format_chat_history(messages, max_turns=8) -> str:
    """Keep a short rolling conversation history (manual memory)."""
    recent = messages[-(max_turns * 2):] if messages else []
    lines = []
    for m in recent:
        if isinstance(m, HumanMessage):
            lines.append(f"User: {m.content}")
        elif isinstance(m, AIMessage):
            lines.append(f"Assistant: {m.content}")
    return "\n".join(lines)


# -------------------------
# STREAMLIT APP
# -------------------------
def main():
    load_dotenv()

    st.set_page_config(page_title="Chat with CSV (Latest)", page_icon="📊")
    st.header("Chat with CSV Data 📊 (Latest LangChain, No Chains)")

    if not os.getenv("OPENAI_API_KEY"):
        st.error("OPENAI_API_KEY not found. Put it in .env or system environment variables.")
        st.stop()

    # Session init
    if "vectorstore" not in st.session_state:
        st.session_state.vectorstore = None
    if "messages" not in st.session_state:
        st.session_state.messages = []  # HumanMessage / AIMessage
    if "csv_name" not in st.session_state:
        st.session_state.csv_name = None

    # Sidebar upload/process
    with st.sidebar:
        st.subheader("Upload CSV")
        csv_file = st.file_uploader("Upload your CSV file", type=["csv"])

        if st.button("Process") and csv_file:
            with st.spinner("Processing CSV..."):
                df = pd.read_csv(csv_file)
                raw_text = csv_to_text(df)
                chunks = get_text_chunks(raw_text)

                st.session_state.vectorstore = build_vectorstore(chunks)
                st.session_state.csv_name = csv_file.name
                st.session_state.messages = []  # reset chat on new CSV

            st.success(f"CSV processed ✅ (chunks: {len(chunks)})")

        if st.button("Rebuild Vector DB"):
            st.cache_resource.clear()
            st.session_state.vectorstore = None
            st.session_state.csv_name = None
            st.session_state.messages = []
            st.success("Cache cleared. Upload and Process again.")

        if st.session_state.csv_name:
            st.caption(f"Active CSV: {st.session_state.csv_name}")

    # Question box
    user_question = st.text_input("Ask a question about your CSV data")

    if user_question:
        if st.session_state.vectorstore is None:
            st.warning("Upload a CSV and click Process first.")
            st.stop()

        retriever = st.session_state.vectorstore.as_retriever(search_kwargs={"k": 5})
        llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

        with st.spinner("Searching CSV knowledge base..."):
            # 1) retrieve relevant rows/chunks
            docs = retriever.invoke(user_question)
            context = "\n\n".join([d.page_content for d in docs])

            # 2) include recent chat history (manual memory)
            history_text = format_chat_history(st.session_state.messages)

            # 3) build messages
            messages = [
                SystemMessage(
                    content=(
                        "You are a helpful assistant that answers questions about CSV data.\n"
                        "Use ONLY the provided context.\n"
                        "If the answer is not in the context, say: "
                        "'I don’t know based on the CSV data.'"
                    )
                ),
                HumanMessage(
                    content=(
                        f"Chat history:\n{history_text}\n\n"
                        f"Context (CSV rows/chunks):\n{context}\n\n"
                        f"Question: {user_question}"
                    )
                )
            ]

            response = llm.invoke(messages)

        # store conversation
        st.session_state.messages.append(HumanMessage(content=user_question))
        st.session_state.messages.append(AIMessage(content=response.content))

        # display conversation
        for m in st.session_state.messages:
            if isinstance(m, HumanMessage):
                st.markdown(f"**You:** {m.content}")
            else:
                st.markdown(f"**Bot:** {m.content}")

        # optional: show sources
        with st.expander("View retrieved CSV chunks (sources)"):
            for i, d in enumerate(docs, start=1):
                st.write(f"Chunk {i}:")
                st.write(d.page_content)
                st.write("---")


if __name__ == "__main__":
    main()
    