FROM python:3.11-slim

WORKDIR /app

# Install Python deps first 
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Pre-download the embedding model so cold start is faster
RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('all-MiniLM-L6-v2')"

# Copy app
COPY . .

# HF Spaces runs as non-root user 1000
RUN useradd -m -u 1000 appuser && chown -R appuser /app
USER appuser

EXPOSE 7860
ENV EPIRAG_ENV=cloud

CMD ["python", "server.py"]