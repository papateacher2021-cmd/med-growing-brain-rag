import sys
try:
    __import__('pysqlite3')
    sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')
except ImportError:
    pass

import streamlit as st
import os
import pandas as pd
import plotly.express as px
from sklearn.decomposition import PCA
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

# --- 2. INICIALIZAR EMBEDDINGS ---
@st.cache_resource
def get_embeddings():
    return HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

embeddings = get_embeddings()

# --- 3. UI CONFIG ---
st.set_page_config(page_title="F. Broissin Marine Equipment Directive RAG", page_icon="🧠", layout="wide")
st.title("🧠 Pancho's MED RAG (Advanced Edition)")

# --- 4. GESTIÓN DE ESTADO (HISTORIAL) ---
if "messages" not in st.session_state:
    st.session_state.messages = []

# --- 5. FUNCIONES DE APOYO ---
def get_indexed_files(vector_db):
    try:
        data = vector_db.get()
        return sorted(list(set([m['source'].split('/')[-1] for m in data['metadatas']])))
    except:
        return []

def plot_3d_space(vector_db):
    try:
        data = vector_db.get(include=['embeddings', 'metadatas', 'documents'])
        if not data['embeddings'] or len(data['embeddings']) < 3:
            return None
        
        pca = PCA(n_components=3)
        vis_dims = pca.fit_transform(data['embeddings'])
        df = pd.DataFrame(vis_dims, columns=['x', 'y', 'z'])
        df['source'] = [m['source'].split('/')[-1] for m in data['metadatas']]
        df['preview'] = [d[:100] + "..." for d in data['documents']]
        
        fig = px.scatter_3d(
            df, x='x', y='y', z='z', color='source',
            hover_data=['preview'], height=400,
            title="3D Vector Space Projection"
        )
        fig.update_layout(margin=dict(l=0, r=0, b=0, t=30), showlegend=False)
        return fig
    except:
        return None

# --- 6. SIDEBAR (GESTIÓN Y LISTADO) ---
with st.sidebar:
    st.image("wheelmark_info.png", use_container_width=True) 
    
    st.header("📂 Knowledge Management")
    uploaded_file = st.file_uploader("🐬 Loading MED-related document (.pdf) 🚢 ", type="pdf")
    
    if uploaded_file:
        file_path = os.path.join(PDF_VAULT, uploaded_file.name)
        with open(file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        
        if st.button("🔄 Process and Learn"):
            with st.spinner("Integrating Knowledge..."):
                loader = PyPDFLoader(file_path)
                data = loader.load()
                text_splitter = RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=300)
                chunks = text_splitter.split_documents(data)
                Chroma.from_documents(documents=chunks, embedding=embeddings, persist_directory=DB_DIR)
                st.success("Knowledge Integrated!")
                st.rerun()

    st.divider()
    st.header("📚 Library Content")
    if os.path.exists(DB_DIR):
        temp_db = Chroma(persist_directory=DB_DIR, embedding_function=embeddings)
        files = get_indexed_files(temp_db)
        if files:
            for f in files:
                st.caption(f"✅ {f}")
        else:
            st.info("Library is empty.")

# --- 7. INTERFAZ PRINCIPAL (CHAT Y VISUALIZACIÓN) ---
col_chat, col_viz = st.columns([2, 1])

with col_chat:
    # Mostrar historial de mensajes
    for i, msg in enumerate(st.session_state.messages):
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if msg["role"] == "assistant":
                st.button("📋 Copy text", key=f"copy_{i}", on_click=lambda t=msg["content"]: st.write(f'<script>navigator.clipboard.writeText("{t}")</script>', unsafe_allow_html=True))

    # Entrada de usuario
    if user_query := st.chat_input("🛟 Please, place your consultation on MED here in English:...🐧"):
        st.session_state.messages.append({"role": "user", "content": user_query})
        with st.chat_message("user"):
            st.markdown(user_query)

        if os.path.exists(DB_DIR):
            vector_db = Chroma(persist_directory=DB_DIR, embedding_function=embeddings)
            retriever = vector_db.as_retriever(search_kwargs={"k": 12})
            docs = retriever.get_relevant_documents(user_query)
            context_text = "\n\n".join([doc.page_content for doc in docs])

            llm = ChatGoogleGenerativeAI(
                model="gemini-2.5-flash", # Actualizado a versión estable recomendada
                google_api_key=os.getenv("GOOGLE_API_KEY"),
                temperature=0.1,
                streaming=True,
                version="v1",
                convert_system_message_to_human=True
            )

            system_instruction = "You are a professional maritime expert. ALWAYS respond in English. "
            prompt = f"{system_instruction}\n\nContext:\n{context_text}\n\nQuestion:\n{user_query}\n\nDetailed Answer in English:"

            with st.chat_message("assistant"):
                placeholder = st.empty()
                full_response = ""
                for chunk in llm.stream(prompt):
                    full_response += chunk.content
                    placeholder.markdown(full_response + "▌")
                placeholder.markdown(full_response)
                st.session_state.messages.append({"role": "assistant", "content": full_response})
        else:
            st.warning("Please upload a PDF first.")

with col_viz:
    st.subheader("🌐 Vector Space 3D")
    if os.path.exists(DB_DIR):
        fig = plot_3d_space(temp_db)
        if fig:
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Upload more data to see the 3D projection.")
