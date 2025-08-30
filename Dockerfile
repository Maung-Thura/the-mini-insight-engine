FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

# Silence GitPython failing when git CLI is not installed; RAGAS imports git on import
ENV GIT_PYTHON_REFRESH=quiet


WORKDIR /app

# System deps (optional, kept minimal)
RUN apt-get update && apt-get install -y --no-install-recommends build-essential && rm -rf /var/lib/apt/lists/*

COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# Copy only necessary sources
COPY app ./app
COPY knowledge_base ./knowledge_base

# Create data directory for Chroma persistence
RUN mkdir -p /app/data
VOLUME ["/app/data"]

EXPOSE 8080

CMD ["python", "-m", "app.server"]

