# Sistema RAG con Ollama + FastAPI + Streamlit + ChromaDB
Este proyecto implementa un sistema de Recuperación Aumentada con Generación (RAG) que permite realizar preguntas sobre documentos PDF proporcionados por el usuario, utilizando un modelo LLM local.

- **Interfaz de usuario**: desarrollada en Streamlit.

- **Backend**: construido con FastAPI.

- **Almacenamiento vectorial**: los embeddings generados se guardan en ChromaDB.

- **Modelo de lenguaje**: el LLM corre de forma local en Ollama, utilizando llama3 como modelo elegido para la generación de respuestas.

# Requisitos Previos
Asegúrate de tener instalado lo siguiente en tu sistema:

Python 3.10+

Git

Ollama: El software debe estar instalado y el servicio de ollama serve debe estar disponible.

 **Modelos de Ollama Requeridos**
Antes de ejecutar el proyecto, descarga los siguientes modelos de Ollama (puedes ejecutar estos comandos en cualquier terminal):

ollama pull llama3         # Para la generación de texto (LLM)
ollama pull all-minilm    # Para generar los embeddings

# Instalación
Sigue estos pasos para configurar tu entorno local

- Clonar el Repositorio:

git clone [TU_URL_DE_GITHUB] RAG
cd RAG

- Crear y Activar el Entorno Virtual:

python -m venv venv
.\venv\Scripts\activate  # En Windows/PowerShell , en Linux o macOS source venv/bin/activate 

- Instalar Dependencias:
python -m pip install -r requirements.txt


# Ejecución del Sistema
El sistema requiere que los tres servidores se ejecuten simultáneamente en terminales separadas

## Terminal 1: Iniciar el Motor de Ollama:
 ollama serve

## Terminal 2: Iniciar el Backend API (FastAPI)
En una nueva terminal (con (venv) activo):

python -m uvicorn main:app --reload

El **backend** estará disponible en http://127.0.0.1:8000


## Terminal 3: Iniciar el Frontend (Streamlit)
En una tercera terminal (con (venv) activo):

streamlit run frontend.py

El **frontend** se abrirá automáticamente en tu navegador (normalmente en http://localhost:8501).

# Guía de Uso
Indexar Documentos: En la interfaz de Streamlit (Puerto 8501), utiliza la sección de carga para subir documentos PDF. Puedes arrastrar o presionar y seleccionar el PDF deseado, hecho esto presionamos el boton **Indexar PDF**. El sistema particionará en chunks, calculará automáticamente los embeddings y los guardará en chroma_db/.

Documentos Indexados: Puedo ver cual archivo se indexo presionando el boton **Actualizar Lista de Documentos**

Chatea con tus Documentos: Escribe tu pregunta en el campo de texto y presiona el boton **consultar**. El sistema consultará la base de datos (ChromaDB) para encontrar el contexto más relevante y lo utilizará para responder con llama3.

Persistencia: La base de datos chroma_db/ persiste en el disco. No es necesario re-indexar los documentos después de reiniciar los servidores.

# Breve explicación de cada archivo y carpeta

- app/ (carpeta)
    Contiene el backend en FastAPI.
    Es la parte que conecta la lógica del RAG con la interfaz y maneja las solicitudes HTTP.

- .gitignore
    Define qué archivos o carpetas no deben subirse al repositorio.

- frontend.py
    Implementa la interfaz en Streamlit.
    Es la capa visual del sistema, pensada para interacción sencilla.

- main.py
    Archivo de entrada principal del proyecto.

- requirements.txt
    Lista de dependencias necesarias para instalar el proyecto.