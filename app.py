import streamlit as st
import os
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI 
from langchain_community.vectorstores import Chroma
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.chains import RetrievalQA

# --- 1. CONFIGURACIÓN DE RUTAS PERSISTENTES ---
# Estas carpetas deben estar en el Mount Path de tu Render Disk
DB_DIR = "/data/chroma_db_med"
PDF_VAULT = "/data/med_pdf_storage"

for folder in [DB_DIR, PDF_VAULT]:
    if not os.path.exists(folder):
        os.makedirs(folder, exist_ok=True)

# --- 2. INTERFAZ DE STREAMLIT ---
st.set_page_config(page_title="MED Growing Brain", page_icon="🧠", layout="wide")
st.title("🧠 MED Growing Brain RAG")
st.markdown("### Marine Equipment Directive - Technical Knowledge Base")

# --- 3. CONFIGURACIÓN ÚNICA DE EMBEDDINGS ---
# Definimos esto fuera para no repetir código y asegurar consistencia
embeddings = GoogleGenerativeAIEmbeddings(
    model="text-embedding-004",  # Sin el prefijo models/ para evitar conflictos en v1
    google_api_version="v1",
    task_type="retrieval_document"
)

# --- 4. SIDEBAR: GESTIÓN DE CONOCIMIENTO ---
with st.sidebar:
    st.header("📥 Knowledge Intake")
    uploaded_file = st.file_uploader("Upload MED Directive (PDF)", type="pdf")
    
    if uploaded_file:
        file_path = os.path.join(PDF_VAULT, uploaded_file.name)
        with open(file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        st.success(f"File '{uploaded_file.name}' secured in the vault.")
        
        if st.button("🔄 Process & Learn"):
            with st.spinner("Expanding the brain..."):
                try:
                    loader = PyPDFLoader(file_path)
                    data = loader.load()
                    # Parámetros exigentes: 2000/300
                    text_splitter = RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=300)
                    chunks = text_splitter.split_documents(data)
                    
                    # Creación/Actualización de la base de datos
                    vector_db = Chroma.from_documents(
                        documents=chunks, 
                        embedding=embeddings, 
                        persist_directory=DB_DIR
                    )
                    st.balloons()
                    st.success("New knowledge integrated successfully!")
                except Exception as e:
                    st.error(f"Error processing PDF: {e}")

# --- 5. MOTOR DE CONSULTA ---
# Verificamos si existe la base de datos buscando el archivo de SQLite
if os.path.exists(os.path.join(DB_DIR, "chroma.sqlite3")):
    user_query = st.text_input("Enter your technical inquiry regarding MED:")
    
    if user_query:
        try:
            vector_db = Chroma(persist_directory=DB_DIR, embedding_function=embeddings)
            
            # Forzamos v1 también en el chat para consistencia total
            llm = ChatGoogleGenerativeAI(
                model="gemini-1.5-flash", 
                temperature=0.1,
                google_api_version="v1"
            )
            
            # K=12 para máxima profundidad
            qa_chain = RetrievalQA.from_chain_type(
                llm=llm, 
                chain_type="stuff", 
                retriever=vector_db.as_retriever(search_kwargs={"k": 12})
            )
            
            with st.spinner("Retrieving facts..."):
                response = qa_chain.invoke(user_query)
                st.markdown("### ⚓ Official Response:")
                st.info(response["result"])
        except Exception as e:
            st.error(f"Error during inquiry: {e}")
else:
    st.warning("The brain is currently empty. Please upload an MED document in the sidebar to begin.")
