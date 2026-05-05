import sys
try:
    __import__('pysqlite3')
    sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')
except ImportError:
    pass

import streamlit as st
import os
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.embeddings import HuggingFaceEmbeddings # <--- La solución
from langchain_community.vectorstores import Chroma
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.chains import RetrievalQA

# --- 1. CONFIGURACIÓN DE RUTAS ---
DB_DIR = "/data/chroma_db_med"
PDF_VAULT = "/data/med_pdf_storage"

for folder in [DB_DIR, PDF_VAULT]:
    os.makedirs(folder, exist_ok=True)

# --- 2. INICIALIZAR EMBEDDINGS (LOCALES) ---
# Al usar HuggingFace, nos olvidamos de los errores de Google API en esta fase.
@st.cache_resource
def get_embeddings():
    return HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

embeddings = get_embeddings()

# --- 3. UI ---
st.set_page_config(page_title="MED Growing Brain", page_icon="🧠", layout="wide")
st.title("🧠 MED Growing Brain (Versión Estable)")

# --- 4. GESTIÓN DE PDFS ---
with st.sidebar:
    st.header("📥 Ingesta de Conocimiento")
    uploaded_file = st.file_uploader("Cargar PDF de la Directiva", type="pdf")
    
    if uploaded_file:
        file_path = os.path.join(PDF_VAULT, uploaded_file.name)
        with open(file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        
        if st.button("🔄 Procesar y Aprender"):
            with st.spinner("Integrando conocimiento localmente..."):
                try:
                    loader = PyPDFLoader(file_path)
                    data = loader.load()
                    
                    # Usamos tu configuración de segmentación que funcionó bien
                    text_splitter = RecursiveCharacterTextSplitter(
                        chunk_size=2000, 
                        chunk_overlap=300,
                        separators=["\nArticle ", "\nARTICLE ", "\n\n", "\n", " ", ""]
                    )
                    chunks = text_splitter.split_documents(data)
                    
                    # Añadir a la base de datos persistente
                    vector_db = Chroma.from_documents(
                        documents=chunks, 
                        embedding=embeddings, 
                        persist_directory=DB_DIR
                    )
                    st.success("✅ ¡Conocimiento integrado!")
                    st.balloons()
                except Exception as e:
                    st.error(f"Error: {e}")

# --- 5. CONSULTA ---
if os.path.exists(os.path.join(DB_DIR, "chroma.sqlite3")):
    user_query = st.text_input("Haz tu consulta técnica sobre MED:")
    
    if user_query:
        try:
            vector_db = Chroma(persist_directory=DB_DIR, embedding_function=embeddings)
            
            # Usamos Gemini para la parte de "razonamiento"
            llm = ChatGoogleGenerativeAI(
                model="gemini-1.5-flash", # Puedes probar 2.0-flash si prefieres
                temperature=0.1
            )
            
            qa_chain = RetrievalQA.from_chain_type(
                llm=llm, 
                chain_type="stuff", 
                retriever=vector_db.as_retriever(search_kwargs={"k": 12})
            )
            
            with st.spinner("Consultando el cerebro persistente..."):
                response = qa_chain.invoke(user_query)
                st.info(response["result"])
        except Exception as e:
            st.error(f"Error en consulta: {e}")
else:
    st.warning("El cerebro está vacío. Sube un PDF para comenzar.")
