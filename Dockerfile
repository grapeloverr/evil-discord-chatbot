# Use Python 3.11 slim image
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \\
    gcc \\
    g++ \\
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Create non-root user for security
RUN useradd --create-home --shell /bin/bash bot \\
    && chown -R bot:bot /app
USER bot

# Create directories for data
RUN mkdir -p /app/data /app/logs

# Environment variables
ENV PYTHONPATH=/app/src
ENV BOT_ENV=production

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \\
    CMD python -c "import sys; sys.exit(0)"

# Expose port (if needed for web interface)
EXPOSE 8080

# Default command
CMD ["python", "main.py"]
