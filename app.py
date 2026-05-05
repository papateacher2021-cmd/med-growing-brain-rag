import sys
try:
    __import__('pysqlite3')
    sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')
except ImportError:
    pass

import streamlit as st
import os
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

# --- 1. CONFIGURACIÓN DE RUTAS ---
DB_DIR = "/data/chroma_db_med"
PDF_VAULT = "/data/med_pdf_storage"

for folder in [DB_DIR, PDF_VAULT]:
    os.makedirs(folder, exist_ok=True)

# --- 2. INICIALIZAR EMBEDDINGS (LOCALES) ---
@st.cache_resource
def get_embeddings():
    return HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

embeddings = get_embeddings()

# --- 3. UI ---
st.set_page_config(page_title="F. Broissin Marine Equipment Directive RAG", page_icon="🧠", layout="wide")
st.title("🧠 Pancho's MED RAG (Versión Streaming)")

# --- 4. GESTIÓN DE PDFS (SIDEBAR) ---
with st.sidebar:
# --- NUEVA LÍNEA PARA LA IMAGEN ---
    st.image("wheelmark_info.png", use_container_width=True) 
    st.header("📥 Knowledge Ingestion")
    uploaded_file = st.file_uploader("🐬 Loading MED-related document (.pdf) 🚢 ", type="pdf")
    
    if uploaded_file:
        file_path = os.path.join(PDF_VAULT, uploaded_file.name)
        with open(file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        
        if st.button("🔄 Process and Learn"):
            with st.spinner("Integrating Knowledge in my CyberBrain...🧠 "):
                try:
                    loader = PyPDFLoader(file_path)
                    data = loader.load()
                    
                    text_splitter = RecursiveCharacterTextSplitter(
                        chunk_size=2000, 
                        chunk_overlap=300,
                        separators=["\nArticle ", "\nARTICLE ", "\n\n", "\n", " ", ""]
                    )
                    chunks = text_splitter.split_documents(data)
                    
                    vector_db = Chroma.from_documents(
                        documents=chunks, 
                        embedding=embeddings, 
                        persist_directory=DB_DIR
                    )
                    st.success("✅ ¡Conocimiento integrado! ⚓ ")
                    st.balloons()
                except Exception as e:
                    st.error(f"Error: {e}")

# Solo para probar, pongo esto antes del bloque del LLM
if os.getenv("GOOGLE_API_KEY"):
    st.sidebar.success("🔑 API Key detectada")
else:
    st.sidebar.error("❌ API Key no encontrada en Environment Variables")

# --- 5. CONSULTA CON STREAMING ---
if os.path.exists(os.path.join(DB_DIR, "chroma.sqlite3")):
    user_query = st.chat_input("🛟  Please, place your consultation on MED here:...🐧")
    
    if user_query:
        # Mostramos la pregunta del usuario
        with st.chat_message("user"):
            st.markdown(user_query)

        try:
            # Conexión a la base de datos
            vector_db = Chroma(persist_directory=DB_DIR, embedding_function=embeddings)
            
            # Configuración de búsqueda (k=6 para más velocidad)
            retriever = vector_db.as_retriever(search_kwargs={"k": 6})
            docs = retriever.get_relevant_documents(user_query)
            
            # Construir el contexto a partir de los documentos encontrados
            context_text = "\n\n".join([doc.page_content for doc in docs])

            # Configuración de Gemini con Streaming y versión estable v1
            llm = ChatGoogleGenerativeAI(
                model="gemini-1.5-flash",
		google_api_key=os.getenv("GOOGLE_API_KEY"), 
                temperature=0.3,
                streaming=True,
                version="v1" 
            )

            # Prompt para guiar a la IA
            prompt = f"""Responde detalladamente basándote en el contexto proporcionado. 
            Si la respuesta no está en el contexto, di que no está disponible.
            
            Contexto:
            {context_text}
            
            Pregunta:
            {user_query}
            
            Respuesta:"""

            # Lógica de Streaming en la UI
            with st.chat_message("assistant"):
                placeholder = st.empty()
                full_response = ""
                
                # Ejecutamos el flujo palabra por palabra
                for chunk in llm.stream(prompt):
                    full_response += chunk.content
                    placeholder.markdown(full_response + "▌")
                
                placeholder.markdown(full_response)

        except Exception as e:
            st.error(f"Error en consulta: {e}")
else:
    st.warning("My CyberBrain is empty. Please, upload a .PDF to start.")
