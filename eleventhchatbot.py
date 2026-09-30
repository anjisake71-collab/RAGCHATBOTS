import os
import streamlit as st
from dotenv import load_dotenv
from PyPDF2 import PdfReader

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI

from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough, RunnableLambda


# -------------------------
# PDF -> TEXT
# -------------------------
def load_pdfs(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += (page.extract_text() or "") + "\n"
    return text


# -------------------------
# TEXT -> CHUNKS
# -------------------------
def get_text_chunks(text):
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    return splitter.split_text(text)


# -------------------------
# CHUNKS -> CHROMA (persist)
# -------------------------
@st.cache_resource
def get_or_create_vector_store(chunks, persist_dir="./chroma_pdf_lcel"):
    """
    Creates (or loads) a persistent Chroma vector store.
    - If persist_dir exists and has data, it will load it.
    - Otherwise, it will create a new store from chunks.
    """
    embeddings = OpenAIEmbeddings()

    # Load existing persistent DB (if it exists)
    if os.path.exists(persist_dir) and os.listdir(persist_dir):
        vectorstore = Chroma(
            persist_directory=persist_dir,
            embedding_function=embeddings
        )
        return vectorstore, "loaded"

    # Create new DB
    vectorstore = Chroma.from_texts(
        texts=chunks,
        embedding=embeddings,
        persist_directory=persist_dir
    )
    vectorstore.persist()
    return vectorstore, "created"


# -------------------------
# LCEL PIPELINE
# -------------------------
def get_lcel_pipeline(vectorstore):
    retriever = vectorstore.as_retriever(search_kwargs={"k": 5})

    template = """You are a helpful assistant. Use the context below to answer the question.
If the answer is not in the context, say: "I don't know based on the PDF."

Context:
{context}

Question:
{question}
"""
    prompt = ChatPromptTemplate.from_template(template)
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

    # retriever returns a list of Documents -> convert to a single string for {context}
    format_docs = RunnableLambda(lambda docs: "\n\n".join(d.page_content for d in docs))

    rag_chain = (
        {
            "context": retriever | format_docs,
            "question": RunnablePassthrough()
        }
        | prompt
        | llm
    )

    return rag_chain


# -------------------------
# STREAMLIT APP
# -------------------------
def main():
    load_dotenv()

    st.set_page_config("Chat with PDF (LCEL - Latest)", page_icon="📄")
    st.header("Chat with PDF 📄 (Chroma + LCEL) — Latest Style")

    if not os.getenv("OPENAI_API_KEY"):
        st.error("OPENAI_API_KEY not found. Put it in .env or system environment variables.")
        st.stop()

    if "pipeline" not in st.session_state:
        st.session_state.pipeline = None

    user_question = st.text_input("Ask a question from your PDF")
    if user_question and st.session_state.pipeline:
        response = st.session_state.pipeline.invoke(user_question)
        st.markdown(f"**Bot:** {response.content}")

    with st.sidebar:
        st.header("Upload PDF(s)")
        pdf_docs = st.file_uploader("Upload PDFs", accept_multiple_files=True, type=["pdf"])

        if st.button("Process PDFs") and pdf_docs:
            with st.spinner("Processing PDFs..."):
                raw_text = load_pdfs(pdf_docs)
                chunks = get_text_chunks(raw_text)

                vectorstore, status = get_or_create_vector_store(
                    chunks, persist_dir="./chroma_pdf_lcel"
                )

                st.session_state.pipeline = get_lcel_pipeline(vectorstore)

            st.success(f"PDFs processed ✅ (Vector DB {status}, LCEL pipeline ready!)")

        if st.button("Rebuild Vector DB"):
            st.cache_resource.clear()
            st.session_state.pipeline = None
            st.success("Cleared cache. Click Process PDFs again.")


if __name__ == "__main__":
    main()
    