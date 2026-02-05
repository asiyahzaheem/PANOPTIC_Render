FROM python:3.11-slim

WORKDIR /app

# Install PyTorch CPU-only first (much smaller than default CUDA build)
RUN pip install --no-cache-dir torch torchvision --index-url https://download.pytorch.org/whl/cpu

# Install remaining deps
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy code + artifacts
COPY . .

ENV PYTHONUNBUFFERED=1

# Railway/Render provide $PORT
CMD ["bash", "-lc", "uvicorn pdac.api.main:app --host 0.0.0.0 --port ${PORT:-8000}"]
