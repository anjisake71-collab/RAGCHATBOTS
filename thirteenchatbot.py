import os
import streamlit as st
from dotenv import load_dotenv

from langchain_community.document_loaders import WebBaseLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS

from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage


# -------------------------
# Load website content
# -------------------------
def load_website(url):
    loader = WebBaseLoader(url)
    docs = loader.load()
    return docs


# -------------------------
# Split into chunks
# -------------------------
def get_text_chunks(docs):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=100,
        separators=["\n\n", "\n", " ", ""]
    )
    return splitter.split_documents(docs)


# -------------------------
# Create vector store
# -------------------------
@st.cache_resource
def build_vector_store(chunks):
    embeddings = OpenAIEmbeddings()
    vectorstore = FAISS.from_documents(chunks, embedding=embeddings)
    return vectorstore


def format_chat_history(messages, max_turns=8):
    recent = messages[-(max_turns * 2):] if messages else []
    lines = []
    for m in recent:
        if isinstance(m, HumanMessage):
            lines.append(f"User: {m.content}")
        elif isinstance(m, AIMessage):
            lines.append(f"Assistant: {m.content}")
    return "\n".join(lines)


# -------------------------
# Main Streamlit App
# -------------------------
def main():
    load_dotenv()

    st.set_page_config("Chat with Website (Latest)", page_icon="🌐")
    st.header("Chat with Website 🌐 (Latest LangChain, No Chains)")

    if not os.getenv("OPENAI_API_KEY"):
        st.error("OPENAI_API_KEY not found. Put it in .env or environment variables.")
        st.stop()

    # Session init
    if "vectorstore" not in st.session_state:
        st.session_state.vectorstore = None
    if "messages" not in st.session_state:
        st.session_state.messages = []  # HumanMessage / AIMessage
    if "active_url" not in st.session_state:
        st.session_state.active_url = None

    # Sidebar
    with st.sidebar:
        st.subheader("Website Input")
        url = st.text_input(
            "Enter Website URL",
            placeholder="https://en.wikipedia.org/wiki/Generative_artificial_intelligence"
        )

        if st.button("Process Website") and url:
            with st.spinner("Loading and processing website..."):
                docs = load_website(url)
                chunks = get_text_chunks(docs)

                st.session_state.vectorstore = build_vector_store(chunks)
                st.session_state.active_url = url
                st.session_state.messages = []  # reset chat for new website

            st.success(f"Website processed ✅ (chunks: {len(chunks)})")
            st.write("Now ask questions about this website.")

        if st.button("Rebuild Vector DB"):
            st.cache_resource.clear()
            st.session_state.vectorstore = None
            st.session_state.active_url = None
            st.session_state.messages = []
            st.success("Cleared cache. Process the website again.")

    # Question input
    user_question = st.text_input("Ask a question from the website content")

    if user_question:
        if st.session_state.vectorstore is None:
            st.warning("Enter a URL in the sidebar and click Process Website first.")
            st.stop()

        retriever = st.session_state.vectorstore.as_retriever(search_kwargs={"k": 5})
        llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

        with st.spinner("Retrieving content and answering..."):
            docs = retriever.invoke(user_question)
            context = "\n\n".join([d.page_content for d in docs])

            history_text = format_chat_history(st.session_state.messages)

            messages = [
                SystemMessage(
                    content=(
                        "You are a helpful assistant for website-based Q&A.\n"
                        "Use ONLY the provided context.\n"
                        "If the answer is not in the context, say: "
                        "'I don’t know based on the website content.'"
                    )
                ),
                HumanMessage(
                    content=(
                        f"Website: {st.session_state.active_url}\n\n"
                        f"Chat history:\n{history_text}\n\n"
                        f"Context:\n{context}\n\n"
                        f"Question: {user_question}"
                    )
                )
            ]

            response = llm.invoke(messages)

        # store conversation
        st.session_state.messages.append(HumanMessage(content=user_question))
        st.session_state.messages.append(AIMessage(content=response.content))

        # render conversation
        for m in st.session_state.messages:
            if isinstance(m, HumanMessage):
                st.markdown(f"**You:** {m.content}")
            else:
                st.markdown(f"**Bot:** {m.content}")

        # teaching: show retrieved chunks
        with st.expander("View retrieved website chunks (sources)"):
            for i, d in enumerate(docs, start=1):
                st.write(f"Chunk {i}:")
                st.write(d.page_content)
                st.write("---")


if __name__ == "__main__":
    main()
   