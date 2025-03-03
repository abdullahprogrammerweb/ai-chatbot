import streamlit as st
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
import ollama

# Load PDF and split into chunks
def process_pdf(pdf_path):
    loader = PyPDFLoader(pdf_path)
    pages = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    return text_splitter.split_documents(pages)

# Create vector database
def create_vector_db(docs):
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    return FAISS.from_documents(docs, embeddings)

# Generate answer using DeepSeek-R1
def generate_answer(question, context):
    prompt = f"Use this context to answer: {context}\n\nQuestion: {question}"
    response = ollama.chat(model='deepseek-r1:1.5b', messages=[{'role': 'user', 'content': prompt}])
    return response['message']['content']

# Streamlit UI
st.title("PDF Q&A with DeepSeek-R1")
uploaded_file = st.file_uploader("Upload a PDF", type="pdf")

if uploaded_file:
    with open("temp.pdf", "wb") as f:
        f.write(uploaded_file.getbuffer())
    docs = process_pdf("temp.pdf")
    db = create_vector_db(docs)
    
    question = st.text_input("Ask a question:")
    if question:
        relevant_docs = db.similarity_search(question, k=3)
        context = "\n".join([doc.page_content for doc in relevant_docs])
        answer = generate_answer(question, context)
        st.write(answer)