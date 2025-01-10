ARG PYTHON_VERSION=3.12.7
FROM python:${PYTHON_VERSION}-slim as base

WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application
COPY . .

# Expose the frontend port
EXPOSE 8000

# Use Flask development server
CMD ["flask", "run", "--host=0.0.0.0", "--port=8000"]