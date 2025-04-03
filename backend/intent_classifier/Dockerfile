# Use an official lightweight Python image
FROM python:3.10-slim

# Set environment variables for Python
ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONUNBUFFERED 1

# Set working directory in container
WORKDIR /app

# Copy the main requirements file from the parent directory
# Assumes the Docker build context is the 'backend/intent_classifier' directory
COPY ../requirements.txt requirements.txt

# Install system dependencies that might be needed (optional, depends on base image)
# RUN apt-get update && apt-get install -y --no-install-recommends some-package && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
# Using --no-cache-dir reduces image size
RUN pip install --no-cache-dir --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt

# Copy the FastAPI application code
COPY api.py .

# Copy the trained model artifacts
# Assumes 'trained_model' directory exists in the build context (backend/intent_classifier)
COPY trained_model ./trained_model

# Expose the port the app runs on (matching the CMD)
# Cloud Run uses the PORT env var, which defaults to 8080
EXPOSE 8080

# Command to run the Uvicorn server
# It will listen on all interfaces (0.0.0.0) on the specified port
# The port 8080 is commonly used and is the default for Cloud Run
CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8080"] 