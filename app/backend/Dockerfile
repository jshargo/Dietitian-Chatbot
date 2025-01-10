ARG PYTHON_VERSION=3.12.7
FROM python:${PYTHON_VERSION}-slim as base

WORKDIR /app

RUN apt-get update && apt-get install -y \
    build-essential \
    g++ \
    curl \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8001

CMD ["python", "main.py"]