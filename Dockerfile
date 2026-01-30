FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

# Install dependencies
COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt

# App code
COPY . /app

# Persisted data lives at /data (matches StaticConfig.DATABASE_URL),
# and we symlink /app/data -> /data so StaticConfig.DATA_DIR ("data") also lands there.
RUN mkdir -p /data \
    && ln -s /data /app/data

EXPOSE 8000

CMD ["python", "-m", "uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
