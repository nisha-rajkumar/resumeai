import streamlit as st
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
import os
import tempfile

# ---- Page Setup ----
st.set_page_config(page_title="📄 Ask your PDF", page_icon="📄")
st.title("📄 Chat with your PDF using RAG + Memory")

# ---- Upload PDF ----
uploaded_file = st.file_uploader("Upload a PDF", type=["pdf"])

# ---- Question Input ----
user_question = st.text_input("Ask a question about the PDF")

# ---- API Key ----
GOOGLE_API_KEY = "AIzaSyC_iRD_Ss1ayBXadgIIrHAoMKu2xoXkTFY"
os.environ["GOOGLE_API_KEY"] = GOOGLE_API_KEY

# ---- Session State for Chat History ----
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

def process_pdf(file):
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        tmp.write(file.read())
        loader = PyPDFLoader(tmp.name)
        pages = loader.load()
    return pages

def create_vector_store(pages):
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    docs = text_splitter.split_documents(pages)
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    vectorstore = FAISS.from_documents(docs, embeddings)
    return vectorstore

if uploaded_file:
    st.success("✅ PDF uploaded successfully")
    pages = process_pdf(uploaded_file)
    vectorstore = create_vector_store(pages)

    # ---- Setup LLM and Memory ----
    llm = ChatGoogleGenerativeAI(model="models/gemini-2.0-flash", temperature=0)
    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

    qa_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vectorstore.as_retriever(),
        memory=memory
    )

    if user_question:
        with st.spinner("Thinking..."):
            result = qa_chain({"question": user_question})
            st.session_state.chat_history.append((user_question, result["answer"]))

        # ---- Display Chat History ----
        for i, (q, a) in enumerate(st.session_state.chat_history):
            st.markdown(f"**❓ You:** {q}")
            st.markdown(f"**🤖 PDF Bot:** {a}")
