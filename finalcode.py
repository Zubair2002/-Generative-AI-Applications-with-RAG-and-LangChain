import streamlit as st
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.chains import RetrievalQA
from langchain.llms import OpenAI  # You can replace this with another LLM if needed
import os
import tempfile

# Set page title
st.set_page_config(page_title="PDF QA Bot")

st.title("📄 PDF QA Bot")

# Upload PDF
uploaded_file = st.file_uploader("Upload a PDF", type="pdf")

if uploaded_file:
    # Save uploaded file to a temporary location
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
        tmp_file.write(uploaded_file.read())
        pdf_path = tmp_file.name

    # Load PDF using PyPDFLoader
    loader = PyPDFLoader(pdf_path)
    documents = loader.load()

    # Split documents into chunks
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = splitter.split_documents(documents)

    # Use HuggingFace Embeddings
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

    # Create vectorstore using Chroma
    vectordb = Chroma.from_documents(chunks, embedding=embeddings)

    # Create retriever
    retriever = vectordb.as_retriever()

    # Use any LLM (e.g., OpenAI – if key is set, otherwise replace this part with another LLM)
    llm = OpenAI(temperature=0, openai_api_key=) # Set your own OPENAI_API_KEY 

    # Create QA chain
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=retriever,
        return_source_documents=True
    )

    # User query input
    query = st.text_input("Ask a question about the document:", value="What this paper is talking about?")

    if st.button("Ask"):
        with st.spinner("Generating answer..."):
            result = qa_chain({"query": query})
            st.subheader("Answer:")
            st.write(result["result"])

            st.subheader("Sources:")
            for doc in result["source_documents"]:
                st.write(doc.metadata["source"])
