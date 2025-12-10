# rag/app/services/rag_pipeline.py

import os
import tempfile
from fastapi import UploadFile

# Importaciones de LangChain para RAG (Indexación)
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.vectorstores import Chroma

# Importaciones de LangChain para RAG (Consulta)
from langchain_community.llms import Ollama
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate

# --- Configuración de Modelos y Rutas ---
OLLAMA_EMBEDDING_MODEL = "all-minilm" # Usado para generar vectores
OLLAMA_LLM_MODEL = "llama3"                # Usado para generar la respuesta
CHROMA_PATH = "chroma_db"                  # Carpeta de la base de datos
# ----------------------------------------


def get_ollama_embeddings() -> OllamaEmbeddings:
    """Inicializa y retorna el objeto de embeddings de Ollama."""
    return OllamaEmbeddings(model=OLLAMA_EMBEDDING_MODEL)

def index_document(uploaded_file: UploadFile) -> str:
    """
    Procesa un documento subido: lo carga, lo fragmenta, genera embeddings
    y lo almacena en ChromaDB.
    """
    # 1. Almacenamiento Temporal
    # ----------------------------
    with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
        tmp_file.write(uploaded_file.file.read())
        temp_file_path = tmp_file.name

    try:
        # 2. Carga del Documento (Document Loading)
        # ----------------------------------------
        loader = PyPDFLoader(temp_file_path)
        documents = loader.load()

        # 3. Fragmentación del Texto (Chunking)
        # --------------------------------------
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=800,
            chunk_overlap=200,
            length_function=len,
            is_separator_regex=False,
        )
        chunks = text_splitter.split_documents(documents)

        # 4. Generación y Almacenamiento de Vectores (Embeddings & Vector Store)
        # ----------------------------------------------------------------------
        embeddings = get_ollama_embeddings()
        
        Chroma.from_documents(
            documents=chunks,
            embedding=embeddings,
            persist_directory=CHROMA_PATH,
        )

        return f"Indexación completa. Se han procesado {len(chunks)} fragmentos."

    finally:
        # 5. Limpieza
        # -------------
        os.remove(temp_file_path)

# ====================================================================
# NUEVA FUNCIÓN: Consulta (Query)
# ====================================================================

def query_document(question: str) -> str:
    """
    Realiza una consulta a la base de datos vectorial y usa el LLM
    para generar una respuesta basada en el contexto recuperado.
    """
    # 1. Cargar el Vector Store y el Retriever
    # ------------------------------------------
    embeddings = get_ollama_embeddings()
    
    # Carga la base de datos de ChromaDB desde el disco.
    vector_store = Chroma(
        persist_directory=CHROMA_PATH,
        embedding_function=embeddings
    )
    
    # Convierte el Vector Store en un Retriever.
    # El Retriever busca los fragmentos más relevantes para la pregunta.
    retriever = vector_store.as_retriever(search_type="mmr", search_kwargs={"k":10, "fetch_k":30})

    # La llamada a as_retriever() con search_type="mmr" le dice al 
    # Retriever que use el algoritmo de MMR.
    # search_kwargs={"k": 3} le dice al Retriever que recupere los 3 
    # fragmentos de texto más relevantes.

    # 2. Inicializar el LLM de Ollama
    # ---------------------------------
    # Esto conecta a tu modelo llama3 instalado localmente.
    llm = Ollama(model=OLLAMA_LLM_MODEL)

# 3. CREAR LA PLANTILLA DE PROMPT CON INSTRUCCIÓN EN ESPAÑOL
    # --------------------------------------------------------
    template = """
    Eres un asistente experto en el análisis de documentos. Tu objetivo es responder a la pregunta del usuario utilizando la información del CONTEXTO.
    
    Si el contexto proporciona información que permite responder la pregunta, úsala. Si la información no se relaciona, explica qué información falta o qué parte del documento deberías revisar.
    
    Responde siempre en español.

    Contexto: {context}
    Pregunta: {question}
    Respuesta en español:"""

    QA_PROMPT = PromptTemplate(
        template=template, input_variables=["context", "question"]
    )


    # 4. Crear la Cadena RAG (RetrievalQA)
    # ------------------------------------
    # RetrievalQA combina 3 pasos:
    # a) Pasa la 'question' al 'retriever' -> obtiene los fragmentos más relevantes (contexto).
    # b) Envía una prompt al 'llm' con la 'question' + 'contexto' (llama al LLM).
    # c) Obtiene la respuesta final.
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
         chain_type="map_reduce", # map_reduce es bueno para respuestas más largas y detalladas
        retriever=retriever,
        return_source_documents=False, # Opcional: poner a True para ver qué fragmentos se usaron
        chain_type_kwargs={"prompt": QA_PROMPT}
    )
    
    # 5. Ejecutar la Cadena
    # ----------------------
    result = qa_chain.invoke({"query": question})
    print(result["source_documents"]) #para debug
    
    # La respuesta final está en la clave 'result' del diccionario retornado.
    return result["result"]