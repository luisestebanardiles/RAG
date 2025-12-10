import streamlit as st
import requests
import json
import os # Necesario para obtener el nombre del host/puerto

# --- CONFIGURACIÓN ---
# Asumimos que FastAPI se ejecuta en 127.0.0.1:8000
FASTAPI_HOST = "http://127.0.0.1:8000" 

# --- FUNCIÓN DE UTILIDAD ---
def get_api_url(endpoint):
    return f"{FASTAPI_HOST}/api/v1{endpoint}"

# --- DISEÑO DE LA PÁGINA ---
st.set_page_config(page_title="RAG Chatbot", layout="wide")
st.title("Asistente RAG con Llama3 y FastAPI")

# ====================================================================
# SECCIÓN 1: SUBIR DOCUMENTOS (Endpoint POST /documents/upload)
# ====================================================================
st.header("1. Subir Documento para Indexar")

uploaded_file = st.file_uploader(
    "Selecciona un archivo PDF para indexar:",
    type=["pdf"],
    accept_multiple_files=False,
    key="file_uploader"
)

if st.button("Indexar PDF", key="index_button") and uploaded_file is not None:
    st.info("Iniciando indexación... Esto puede tardar unos minutos.")
    
    # Preparamos la solicitud multipart/form-data
    files = {'file': (uploaded_file.name, uploaded_file.getvalue(), 'application/pdf')}
    
    try:
        response = requests.post(get_api_url("/documents/upload"), files=files, timeout=300)
        
        if response.status_code == 202:
            st.success("✅ Documento indexado con éxito.")
        else:
            st.error(f"❌ Error en la indexación: {response.status_code}")
            st.json(response.json())
            
    except requests.exceptions.RequestException as e:
        st.error(f"❌ Error de conexión con el servidor FastAPI: {e}")

st.divider()

# ====================================================================
# SECCIÓN 2: LISTAR DOCUMENTOS (Endpoint GET /documents/list)
# ====================================================================
st.header("2. Documentos Indexados")

if st.button("Actualizar Lista de Documentos", key="list_button"):
    try:
        response = requests.get(get_api_url("/documents/list"))
        
        if response.status_code == 200:
            doc_list = response.json()
            if doc_list:
                st.write("Archivos disponibles para consulta:")
                # Creamos una lista enumerada con Markdown
                st.markdown("\n".join([f"* {doc}" for doc in doc_list]))
            else:
                st.info("Aún no hay documentos indexados. Por favor, suba un PDF.")
        else:
            st.error(f"❌ Error al obtener la lista: {response.status_code}")
            
    except requests.exceptions.RequestException:
        st.error("❌ No se pudo conectar con el servidor FastAPI.")

st.divider()

# ====================================================================
# SECCIÓN 3: CONSULTAS RAG (Endpoint POST /query)
# ====================================================================
st.header("3. Chatea con tus Documentos")

query = st.text_area("Escribe tu pregunta:", height=100, key="query_input")

if st.button("Consultar", key="query_button") and query:
    st.info("Consultando al modelo RAG...")
    
    # Preparamos la solicitud JSON
    data = {"question": query}
    
    try:
        response = requests.post(get_api_url("/query"), json=data, timeout=300)

        if response.status_code == 200:
            result = response.json()
            st.subheader("Respuesta del Asistente")
            # Mostrar la respuesta final de forma elegante
            st.markdown(
                f'<div style="background-color:#0e1117; padding: 15px; border-radius: 5px; border-left: 5px solid #00f3c5; white-space: pre-wrap;">{result["answer"]}</div>',
                unsafe_allow_html=True
            )
            # Opcional: Mostrar el modelo usado
            st.caption(f"Modelo LLM utilizado: {result.get('llm_model', 'N/A')}")
            
        else:
            st.error(f"❌ Error al procesar la consulta: {response.status_code}")
            st.json(response.json())
            
    except requests.exceptions.RequestException as e:
        st.error(f"❌ Error de conexión o timeout con el servidor FastAPI: {e}")