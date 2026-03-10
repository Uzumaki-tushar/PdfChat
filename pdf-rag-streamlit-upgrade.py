import streamlit as st
import os
import tempfile
# from langchain_community.document_loaders import UnstructuredPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings # No local Ollama needed
from langchain_groq import ChatGroq # Cloud-based LLM
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_community.document_loaders import PyPDFLoader

# --- CONFIGURATION ---
# Set your API Key here or in Streamlit Secrets
GROQ_API_KEY = st.secrets["GROQ_API_KEY"]

def ingest_pdf(uploaded_file):
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
        tmp_file.write(uploaded_file.getvalue())
        tmp_path = tmp_file.name

    loader = PyPDFLoader(tmp_path)
    data = loader.load()

    os.remove(tmp_path)
    return data

@st.cache_resource
def get_vector_db(_chunks):
    # This runs on the CPU, no GPU needed for hosting
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    return Chroma.from_documents(documents=_chunks, embedding=embeddings)

def main():
    st.title("🚀 Chat with your PDF")
    st.text("open the sidebar and input the your PDF to start the chat.......")
    
    uploaded_file = st.sidebar.file_uploader("Upload PDF", type="pdf")

    if uploaded_file:
        if "vector_db" not in st.session_state:
            with st.spinner("Indexing PDF..."):
                docs = ingest_pdf(uploaded_file)
                chunks = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200).split_documents(docs)
                st.session_state.vector_db = get_vector_db(chunks)

        user_input = st.text_input("Ask anything about the document:")

        if user_input:
            # Initialize Cloud LLM
            llm = ChatGroq(groq_api_key=GROQ_API_KEY, model_name="llama-3.3-70b-versatile")
            
            # Simple RAG Chain
            retriever = st.session_state.vector_db.as_retriever()
            template = "Answer based on context: {context}\n\nQuestion: {question}"
            prompt = ChatPromptTemplate.from_template(template)
            
            chain = (
                {"context": retriever, "question": RunnablePassthrough()}
                | prompt
                | llm
                | StrOutputParser()
            )
            
            st.write(chain.invoke(user_input))

if __name__ == "__main__":
    main()