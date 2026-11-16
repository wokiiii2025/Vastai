FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    IMAGE_TASK_DB=/data/tasks.sqlite3 \
    COMFY_INPUT_DIR=/data/inputs \
    COMFY_OUTPUT_DIR=/data/output

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt \
    && groupadd --gid 10001 imageapi \
    && useradd --uid 10001 --gid imageapi --no-create-home imageapi \
    && install -d -o imageapi -g imageapi /data /data/inputs /data/output

COPY openai_images_api.py comfy_backend.py ./
USER imageapi
EXPOSE 8080
CMD ["python", "-m", "uvicorn", "openai_images_api:app", "--host", "0.0.0.0", "--port", "8080", "--workers", "1"]
