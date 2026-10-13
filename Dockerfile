FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    HF_HOME=/app/.cache/huggingface

# OpenCV runtime libraries
RUN apt-get update \
    && apt-get install -y --no-install-recommends libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# CPU-only PyTorch: the VPS has no GPU and the CUDA build adds several GB
COPY backend/requirements.txt backend/requirements.txt
RUN pip install torch==2.14.1 torchvision==0.29.1 --index-url https://download.pytorch.org/whl/cpu \
    && pip install -r backend/requirements.txt

# Bake CLIP into the image so the container starts without downloading ~600 MB
RUN python -c "from transformers import CLIPModel, CLIPProcessor; \
CLIPModel.from_pretrained('openai/clip-vit-base-patch32'); \
CLIPProcessor.from_pretrained('openai/clip-vit-base-patch32')"
ENV HF_HUB_OFFLINE=1

COPY backend backend
COPY frontend frontend

# Fail the build if the model arrived as a Git LFS pointer instead of the real weights
RUN python -c "import os, sys; \
size = os.path.getsize('backend/models/model_final_resnet50.pth'); \
sys.exit(0 if size > 1_000_000 else 'ERROR: model_final_resnet50.pth pesa %d bytes: es un puntero de Git LFS, no el modelo. El clon no descargo los archivos LFS.' % size)"

RUN useradd --create-home appuser && chown -R appuser /app
USER appuser

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=10s --start-period=120s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/', timeout=5)" || exit 1

CMD ["uvicorn", "backend.app.main:app", "--host", "0.0.0.0", "--port", "8000"]
