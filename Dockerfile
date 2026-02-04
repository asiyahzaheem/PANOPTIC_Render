FROM python:3.11-slim

WORKDIR /app

# Install deps
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy code + artifacts
COPY . .

ENV PYTHONUNBUFFERED=1

# Render provides $PORT
CMD ["bash", "-lc", "uvicorn pdac.api.main:app --host 0.0.0.0 --port ${PORT:-8000}"]

