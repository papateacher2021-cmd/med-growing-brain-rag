import streamlit as st
import os
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_community.vectorstores import Chroma
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import RetrievalQA

# --- 1. PERSISTENT PATH CONFIGURATION ---
# These folders are mapped to the Render Disk
DB_DIR = "/data/chroma_db_med"
PDF_VAULT = "/data/med_pdf_storage"

# Ensure directories exist in the persistent volume
for folder in [DB_DIR, PDF_VAULT]:
    if not os.path.exists(folder):
        os.makedirs(folder)

# --- 2. STREAMLIT UI ---
st.set_page_config(page_title="MED Growing Brain", page_icon="🧠", layout="wide")
st.title("🧠 MED Growing Brain RAG")
st.markdown("### Marine Equipment Directive - Technical Knowledge Base")

# --- 3. SIDEBAR: DOCUMENT MANAGEMENT ---
with st.sidebar:
    st.header("📥 Knowledge Intake")
    uploaded_file = st.file_uploader("Upload MED Directive (PDF)", type="pdf")
    
    if uploaded_file:
        # Save PDF to persistent disk
        file_path = os.path.join(PDF_VAULT, uploaded_file.name)
        with open(file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        st.success(f"File '{uploaded_file.name}' secured in the vault.")
        
        if st.button("🔄 Process & Learn"):
            with st.spinner("Expanding the brain..."):
                # Load and split
                loader = PyPDFLoader(file_path)
                data = loader.load()
                text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
                chunks = text_splitter.split_documents(data)
                
                # Update persistent vector store
                embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
                Chroma.from_documents(
                    documents=chunks, 
                    embedding=embeddings, 
                    persist_directory=DB_DIR
                )
                st.balloons()
                st.success("New knowledge integrated successfully!")

# --- 4. CONSULTATION ENGINE ---
# Check if the database has content
if os.path.exists(os.path.join(DB_DIR, "chroma.sqlite3")):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    vector_db = Chroma(persist_directory=DB_DIR, embedding_function=embeddings)
    
    user_query = st.text_input("Enter your technical inquiry regarding MED:")
    
    if user_query:
        llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0.1)
        qa_chain = RetrievalQA.from_chain_type(
            llm=llm, 
            chain_type="stuff", 
            retriever=vector_db.as_retriever(search_kwargs={"k": 7})
        )
        
        with st.spinner("Retrieving facts from persistent storage..."):
            response = qa_chain.invoke(user_query)
            st.markdown("### ⚓ Official Response:")
            st.info(response["result"])
else:
    st.warning("The brain is currently empty. Please upload an MED document in the sidebar to begin.")
