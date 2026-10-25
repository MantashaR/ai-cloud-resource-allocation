# Container image for the AI cloud resource allocation dashboard
FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    APP_FILE=app_v2.py

WORKDIR /app

# Install dependencies first so this layer is cached between code changes
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY core/ core/
COPY .streamlit/ .streamlit/
COPY app.py app_v2.py ./

# Run as a non-root user
RUN useradd --create-home appuser && chown -R appuser /app
USER appuser

EXPOSE 8501

HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8501/_stcore/health')"

# sh -c so $APP_FILE is expanded; exec so Streamlit receives stop signals directly
CMD ["sh", "-c", "exec streamlit run \"$APP_FILE\" --server.port=8501 --server.address=0.0.0.0 --server.headless=true --browser.gatherUsageStats=false"]
