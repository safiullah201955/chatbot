import os
import streamlit as st
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain.chains import ConversationalRetrievalChain
import tempfile

# ========= Load API Key =========
load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
if not GOOGLE_API_KEY:
    st.error("Please set GOOGLE_API_KEY in your .env file.")
    st.stop()

# ========= Step 1: Streamlit UI =========
st.set_page_config(page_title="Gemini RAG Chatbot", layout="wide")
st.title("💬 Context-Aware Chatbot with RAG")

# Upload PDF
uploaded_file = st.file_uploader("Upload a knowledge base file (PDF)", type=["pdf"])

# ========= Step 2: Process Document =========
# ========= Step 2: Process Document =========
if uploaded_file:
    with st.spinner("Processing document..."):
        # Save uploaded file to a temporary location
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
            tmp_file.write(uploaded_file.read())
            tmp_file_path = tmp_file.name

        # Load and split
        loader = PyPDFLoader(tmp_file_path)
        docs = loader.load()
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        splits = text_splitter.split_documents(docs)

        # Embeddings + Vector Store
        embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
        vectorstore = FAISS.from_documents(splits, embeddings)

        # ========= Step 3: Setup Gemini Chat Model =========
        llm = ChatGoogleGenerativeAI(model="gemini-pro", temperature=0.2)

        # ========= Step 4: Conversational RAG Chain =========
        qa_chain = ConversationalRetrievalChain.from_llm(
            llm=llm,
            retriever=vectorstore.as_retriever(search_kwargs={"k": 3}),
            return_source_documents=True
        )

        # ========= Step 5: Session Memory =========
        if "chat_history" not in st.session_state:
            st.session_state.chat_history = []

        # User Input
        user_question = st.text_input("Ask me anything from your document:")

        if user_question:
            result = qa_chain({"question": user_question, "chat_history": st.session_state.chat_history})
            answer = result["answer"]

            # Update memory
            st.session_state.chat_history.append((user_question, answer))

            # Display conversation
            for q, a in st.session_state.chat_history:
                st.markdown(f"**You:** {q}")
                st.markdown(f"**Bot:** {a}")
