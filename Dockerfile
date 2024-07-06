FROM python:3.9-slim

WORKDIR /app

COPY setup.py /app/
COPY api /app/api
COPY hsemotion_onnx /app/hsemotion_onnx

RUN apt-get update && apt-get install libgl1 libglib2.0-0 libsm6 libxrender1 libxext6 -y \
    && pip install --upgrade pip \
    && pip install .

CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
