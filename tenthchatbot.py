import os
import streamlit as st
from dotenv import load_dotenv
from PyPDF2 import PdfReader                          

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage


# -------------------------
# PDF -> TEXT
# -------------------------
def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        reader = PdfReader(pdf)
        for page in reader.pages:
            page_text = page.extract_text() or ""
            text += page_text + "\n"
    return text


# -------------------------
# TEXT -> CHUNKS
# -------------------------
def get_text_chunks(text):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=20,
        separators=["\n\n", "\n", " ", ""]
    )
    return splitter.split_text(text)


# -------------------------
# CHUNKS -> VECTORSTORE
# -------------------------
@st.cache_resource
def build_vectorstore_from_chunks(text_chunks):
    embeddings = OpenAIEmbeddings()
    vectorstore = FAISS.from_texts(text_chunks, embedding=embeddings)
    return vectorstore


def format_chat_history(messages, max_turns=8):
    """
    Convert session messages into a short history string
    so the model stays conversational without LangChain memory objects.
    """
    # Keep only recent messages to control prompt size
    recent = messages[-(max_turns * 2):] if messages else []

    lines = []
    for m in recent:
        if isinstance(m, HumanMessage):
            lines.append(f"User: {m.content}")
        elif isinstance(m, AIMessage):
            lines.append(f"Assistant: {m.content}")
    return "\n".join(lines)


# -------------------------
# MAIN APP
# -------------------------
def main():
    load_dotenv()

    st.set_page_config(page_title="Chat with Multiple PDFs (Latest, No Chains)", page_icon="📚")
    st.header("Chat with Multiple PDFs 📚 (Latest LangChain, No Chains)")

    # Key check
    if not os.getenv("OPENAI_API_KEY"):
        st.error("OPENAI_API_KEY not found. Put it in .env or system environment variables.")
        st.stop()

    # Session state init
    if "messages" not in st.session_state:
        st.session_state.messages = []  # stores HumanMessage / AIMessage
    if "vectorstore" not in st.session_state:
        st.session_state.vectorstore = None

    # Sidebar upload/process
    with st.sidebar:
        st.subheader("Your Documents")
        pdf_docs = st.file_uploader(
            "Upload PDF files and click Process",
            accept_multiple_files=True,
            type=["pdf"]
        )

        if st.button("Process"):
            if not pdf_docs:
                st.warning("Please upload at least one PDF.")
            else:
                with st.spinner("Extracting + chunking + building vector DB..."):
                    raw_text = get_pdf_text(pdf_docs)
                    chunks = get_text_chunks(raw_text)
                    st.session_state.vectorstore = build_vectorstore_from_chunks(chunks)
                st.success(f"Processing done ✅  (chunks: {len(chunks)})")

        if st.button("Rebuild Vector DB"):
            st.cache_resource.clear()
            st.session_state.vectorstore = None
            st.success("Cache cleared. Re-upload and Process again.")

    # Input box
    user_question = st.text_input("Ask a question from your documents:")

    if user_question:
        if st.session_state.vectorstore is None:
            st.warning("Upload PDFs and click Process first.")
            st.stop()

        # 1) Retriever from vectorstore
        retriever = st.session_state.vectorstore.as_retriever(search_kwargs={"k": 5})

        with st.spinner("Retrieving relevant chunks and answering..."):
            # 2) Retrieve docs
            docs = retriever.invoke(user_question)
            context = "\n\n".join([d.page_content for d in docs])

            # 3) Prepare conversational prompt manually
            history_text = format_chat_history(st.session_state.messages)

            messages = [
                SystemMessage(
                    content=(
                        "You are a helpful assistant for PDF-based Q&A.\n"
                        "Use ONLY the provided context to answer.\n"
                        "If the answer is not in the context, say: "
                        "'I don’t know based on the uploaded PDFs.'"
                    )
                ),
                HumanMessage(
                    content=(
                        f"Chat history:\n{history_text}\n\n"
                        f"Context:\n{context}\n\n"
                        f"Question: {user_question}"
                    )
                )
            ]

            # 4) Call LLM directly (no chain)
            llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
            response = llm.invoke(messages)

        # 5) Store chat history
        st.session_state.messages.append(HumanMessage(content=user_question))
        st.session_state.messages.append(AIMessage(content=response.content))

        # 6) Render chat (simple)
        for m in st.session_state.messages:
            if isinstance(m, HumanMessage):
                st.markdown(f"**You:** {m.content}")
            else:
                st.markdown(f"**Bot:** {m.content}")

        # Optional: show retrieved chunks for teaching
        with st.expander("View retrieved chunks (sources)"):
            for i, d in enumerate(docs, start=1):
                st.write(f"Chunk {i}:")
                st.write(d.page_content)
                st.write("---")


if __name__ == "__main__":
    main()
    