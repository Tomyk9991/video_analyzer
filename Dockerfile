FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1

# onnxruntime/torch werden bewusst NICHT gebraucht (OpenCV-DNN reicht)
RUN apt-get update && apt-get install -y --no-install-recommends \
    libglib2.0-0 v4l-utils \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY poc/ poc/
# Modell wird beim ersten Start geladen (oder per ENV YOLO_MODEL_PATH gemountet)
RUN mkdir -p models

EXPOSE 8000
CMD ["python", "-m", "poc.main"]
