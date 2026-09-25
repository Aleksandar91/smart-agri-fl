FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /app

COPY requirements-ml.txt .
# CPU wheels only. Default PyPI torch pulls CUDA + nvidia wheels.
RUN pip install --no-cache-dir -r requirements-ml.txt \
    && pip install --no-cache-dir torch==2.13.0 torchvision==0.28.0 \
      --index-url https://download.pytorch.org/whl/cpu
RUN python -c "from torchvision.models import MobileNet_V3_Small_Weights; MobileNet_V3_Small_Weights.DEFAULT.get_state_dict(progress=False)"
COPY app ./app
CMD ["python", "-m", "app.fl_client"]
