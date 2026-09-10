FROM python:3.11-slim AS base

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /app

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
      fswebcam \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt


# Optional target for ML training/inference dependencies (larger image).
# Build it with:
#   docker build --target ml -t fl-client:ml .
FROM base AS ml
COPY requirements-ml.txt .
# CPU wheels only. Default PyPI torch==2.13.0 pulls CUDA + nvidia-* (~GBs) and
# exceeds GitLab shared-runner time; clients here train on CPU (Pi + Docker).
RUN pip install --no-cache-dir -r requirements-ml.txt \
    && pip install --no-cache-dir torch==2.13.0 torchvision==0.28.0 \
      --index-url https://download.pytorch.org/whl/cpu
# Cache pretrained weights once in the image. Four FL clients must not race to
# download the same file at startup, and experiments should start offline.
RUN python -c "from torchvision.models import MobileNet_V3_Small_Weights; MobileNet_V3_Small_Weights.DEFAULT.get_state_dict(progress=False)"
COPY app ./app
CMD ["python", "-m", "app.fl_client"]


# Keep the default build target small (base image).
FROM base AS final
COPY app ./app
CMD ["python", "-m", "app.client"]

