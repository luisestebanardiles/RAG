# Documentación de Decisiones Técnicas Clave

## Arquitectura y Estructura
- **Uso de FastAPI + Uvicorn**: FastAPI fue elegido principalmente por recomendacion (requisito no funcional 1) y uvicorn es el que se usa como complemento

- **Streamlit** como Frontend Se seleccionó para la capa de presentación debido a su rapidez de prototipado con Python, lo que permite implementar el requisito opcional de frontend de manera eficiente y sin la complejidad de configurar CORS o frameworks de JavaScript.

- **Arquitectura de N Capas**: Se eligió para lograr modularidad y claridad (requisito no funcional 2). Separa la Presentación (Streamlit), la Lógica de Negocio (FastAPI) y los Servicios (Ollama/ChromaDB), permitiendo que cada capa sea modificada o escalada independientemente.

## Parámetros y Herramientas del Sistema RAG

- Modelo de LLM: **llama3**, principalmente fue elegido por ser el propuesto pero además es uno de los modelos de código abierto con mejor rendimiento y capacidad de razonamiento, garantizando una alta calidad de respuesta a las consultas.

- Modelo de Embeddings: **all-minilm** Es un modelo compacto y rápido, optimizado para tareas de recuperación de información, lo que minimiza el tiempo de indexación y el consumo de recursos locales. Principalmente se eligió un modelo mas grande y mas potente pero no dio los resultados esperados 

- Base Vectorial: **ChromaDB** Primero se eligió hacer la persistencia local y chromaDB permite tener persistencia local, lo que permite que el sistema recuerde los documentos indexados sin necesidad de re-indexar después de reiniciar el servidor. 

## Configuración y Parámetros del RAG

- Tamaño de Fragmento (Chunk Size=800), este valor fue elegido ya que fue el que mejor resultado tenía ya que un valor mas grande, no traia buenas respuestas y cargaba mucho el sistema

- Solapamiento (Overlap= 200) Se recomienda entre un 10-20%, pero se eligió un valor de solapamiento relativamente alto (25% del chunk size) para mitigar el riesgo de que la información clave o las transiciones de ideas se corten entre fragmentos, ya que fue un problema recurrente. Esto mejora la precisión de la recuperación (Retrieval).


# Herramientas de asistencia

Este proyecto fue desarrollado con el apoyo activo de LLM, principalmente Gemini, ya que luego de consultar con varios modelos, este fue el que mejor se adaptó a mis consultas

La IA ayudó en la generación de fragmentos de código, como asi tambien en depuracion de errores 
Todas las estructuras de código y lógica sugeridas por la IA fueron revisadas, adaptadas e integradas para ajustarse a las clases y funciones preexistentes del proyecto