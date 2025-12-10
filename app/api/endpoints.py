# rag/app/api/endpoints.py
import os
from fastapi import APIRouter, status, File, UploadFile, HTTPException
from pydantic import BaseModel # Importamos para definir el esquema de datos de entrada
from app.services import rag_pipeline

# Creamos un objeto APIRouter.
router = APIRouter()

# ====================================================================
# ESQUEMA DE DATOS (MODELO Pydantic)
# ====================================================================
# Define la estructura de datos que se espera en el cuerpo (body) de la solicitud
class QueryRequest(BaseModel):
    """Esquema para la consulta del usuario."""
    question: str

# ====================================================================
# Endpoint: Consulta (Query)
# ====================================================================

@router.post(
    "/query",
    status_code=status.HTTP_200_OK,
    summary="Realiza una consulta al sistema RAG.",
    tags=["Consulta"]
)
def run_query(request: QueryRequest):
    """
    Envía una pregunta al sistema. El sistema recupera el contexto
    de la base de datos de vectores y usa el LLM (llama3) para
    generar una respuesta contextualizada.
    """
    if not os.path.isdir(rag_pipeline.CHROMA_PATH):
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Base de datos de conocimiento no encontrada. Por favor, indexe un documento primero."
        )

    try:
        # Llamar a la lógica de negocio para la consulta
        response = rag_pipeline.query_document(request.question)
        
        return {
            "question": request.question,
            "answer": response,
            "llm_model": rag_pipeline.OLLAMA_LLM_MODEL
        }
    except Exception as e:
        # Manejo de errores durante la consulta
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error durante la consulta RAG: {e}"
        )


@router.get(
    "/health",
    status_code=status.HTTP_200_OK,
    summary="Verifica el estado del servicio.",
    tags=["General"]
)
def get_health_status():
    """
    Endpoint simple para verificar que el servicio está funcionando.
    """
    return {"status": "ok", "service": "rag-poc-api", "version": "v1"}

@router.post(
    "/documents/upload",
    status_code=status.HTTP_202_ACCEPTED,
    summary="Sube un documento y lo indexa para su consulta.",
    tags=["Documentos"]
)
async def upload_document(file: UploadFile = File(...)):
    """
    Permite al usuario subir un documento (PDF recomendado) al sistema.
    El sistema lo procesará, fragmentará y generará embeddings.
    """
    if file.content_type != "application/pdf":
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Tipo de archivo no soportado: {file.content_type}. Use PDF."
        )

    try:
        message = rag_pipeline.index_document(file)
        
        return {
            "message": message,
            "filename": file.filename,
            "content_type": file.content_type
        }
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error durante la indexación del documento: {e}"
        )