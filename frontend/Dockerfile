FROM python:3.12.7-slim

WORKDIR /app

# Copy requirements first for better caching
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Expose port
EXPOSE 8000

# Add health check and network configuration
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 CMD curl --fail http://localhost:8000/health || exit 1

# Command to run Flask development server
CMD ["flask", "--app", "app", "run", "--host=0.0.0.0", "--port=8000"]