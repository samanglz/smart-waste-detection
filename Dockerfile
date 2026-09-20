FROM python:3.11-slim

WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

COPY requirements-prod.txt .

RUN pip install --no-cache-dir \
    --index-url https://pypi.mirrors.ustc.edu.cn/simple \
    -r requirements-prod.txt

COPY src ./src
COPY exports/E2/best.onnx ./exports/E2/best.onnx

EXPOSE 8000

CMD ["uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000"]