FROM python:3.13-slim

WORKDIR /app

# Copy requirements first to leverage Docker cache
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application
COPY . .

# Create a directory for uploaded files and vector database
RUN mkdir -p /app/uploads /app/vector_db

# Expose port for Streamlit
EXPOSE 8501

# Health check to ensure the app is running
HEALTHCHECK --interval=30s --timeout=3s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8501/healthz || exit 1

# Run the RAG chatbot application
CMD ["streamlit", "run", "rag_chatbot_app.py", "--server.port=8501", "--server.address=0.0.0.0"]