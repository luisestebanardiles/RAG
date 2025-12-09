from fastapi import FastAPI
from app.api import endpoints

# Inicialización de la aplicación FastAPI
app = FastAPI(
    title="RAG PoC System API",
    description="API para la prueba de concepto del sistema RAG con Ollama.",
    version="1.0.0",
)

# Incluir las rutas (endpoints) definidas en nuestro módulo 'app.api.endpoints'
# Esto organiza la aplicación, separando la definición de la aplicación (aquí)
# de la lógica de las rutas (en 'endpoints.py').
app.include_router(endpoints.router, prefix="/api/v1")

# Nota: El servidor Uvicorn se ejecutará con este archivo (main:app).
# Por ejemplo: uvicorn main:app --reload