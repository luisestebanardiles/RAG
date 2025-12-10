from fastapi import FastAPI
from app.api import endpoints

# Inicialización de la aplicación FastAPI
app = FastAPI(
    title="RAG PoC System API",
    description="API para la prueba de concepto del sistema RAG con Ollama.",
    version="1.0.0",
)

# Incluye las rutas de la API
app.include_router(endpoints.router, prefix="/api/v1")

# Nota: El servidor Uvicorn se ejecutará con este archivo (main:app).
# Por ejemplo: uvicorn main:app --reload

# en otra terminal debo ejecutar:
# .\venv\Scripts\activate
#streamlit run frontend.py

# siempre fijarse que ollama este ejecutandose correctamente (ollama serve)