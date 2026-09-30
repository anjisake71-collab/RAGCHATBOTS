import os
import streamlit as st
from dotenv import load_dotenv

from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings, ChatOpenAI

from langchain_core.messages import HumanMessage, SystemMessage

# -------------------------
# ENV
# -------------------------
load_dotenv()

if not os.getenv("OPENAI_API_KEY"):
    st.error("OPENAI_API_KEY not found in environment.")    
    st.stop()

# -------------------------
# UI
# -------------------------
st.set_page_config(page_title="RAG Chatbot - Manual Style", layout="centered")
st.title("📘 RAG Chatbot (Text)")
st.info("Upload a .txt document to build the knowledge base.")

# -------------------------
# Build Vector Store
# -------------------------
@st.cache_resource
def build_vectorstore(uploaded_file):
    temp_path = f"./temp_{uploaded_file.name}"

    with open(temp_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    loader = TextLoader(temp_path, encoding="utf-8")
    documents = loader.load()

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )

    chunks = splitter.split_documents(documents)

    embeddings = OpenAIEmbeddings()
    vectorstore = FAISS.from_documents(chunks, embeddings)

    return vectorstore, len(chunks)


uploaded_file = st.file_uploader("Upload .txt file", type=["txt"])

if not uploaded_file:
    st.stop()

vectorstore, chunk_count = build_vectorstore(uploaded_file)
st.sidebar.success(f"Chunks created: {chunk_count}")

retriever = vectorstore.as_retriever(search_kwargs={"k": 5})

# -------------------------
# LLM
# -------------------------
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

# -------------------------
# Chat
# -------------------------
query = st.text_input("Ask a question about the document:")

if query:
    with st.spinner("Searching document..."):

        # 1️⃣ Retrieve relevant chunks manually
        docs = retriever.invoke(query)

        # 2️⃣ Build context string manually
        context = "\n\n".join([doc.page_content for doc in docs])

        # 3️⃣ Create messages manually (like your earlier bots)
        messages = [
            SystemMessage(
                content="You are a helpful assistant. "
                        "Answer ONLY using the provided context. "
                        "If answer not in context, say you don't know."
            ),
            HumanMessage(
                content=f"Context:\n{context}\n\nQuestion: {query}"
            )
        ]

        # 4️⃣ Call model directly
        response = llm.invoke(messages)

        # 5️⃣ Display answer
        st.success(response.content)

        # Show retrieved chunks (for teaching)
        with st.expander("View retrieved chunks"):
            for i, doc in enumerate(docs, start=1):
                st.write(f"Chunk {i}:")
                st.write(doc.page_content)
                st.write("---")
    