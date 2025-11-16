# orthopredict_app/Dockerfile
FROM python:3.9-slim

# Metadatos para Docker Hub
LABEL org.opencontainers.image.title="OrthoPredict Pro ML"
LABEL org.opencontainers.image.description="Sistema de predicción de ortodoncia con Machine Learning"
LABEL org.opencontainers.image.version="5.2"
LABEL org.opencontainers.image.source="https://github.com/toloporto/orthopredict-app"

# Establecer variables de entorno (MANTENIENDO LAS TUYAS)
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    STREAMLIT_SERVER_PORT=8501 \
    STREAMLIT_SERVER_ADDRESS=0.0.0.0

WORKDIR /app

# Instalar dependencias del sistema (OPTIMIZADO)
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# Copiar requirements e instalar dependencias Python
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copiar código de la aplicación
COPY . .

# Crear directorios necesarios
RUN mkdir -p data models logs backups

# Exponer puerto
EXPOSE 8501

# Health check (COMPATIBLE)
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8501/_stcore/health || exit 1

# Comando para ejecutar la aplicación (MANTENIENDO)
CMD ["streamlit", "run", "src/app.py", \
     "--server.port=8501", \
     "--server.address=0.0.0.0", \
     "--server.headless=true", \
     "--browser.gatherUsageStats=false"]