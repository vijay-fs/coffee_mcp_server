FROM python:3.13.0

# Install required system dependencies
RUN apt-get update -y && \
    apt-get install -y --no-install-recommends \
        tesseract-ocr \
        poppler-utils \
        build-essential \
        libjpeg-dev \
        zlib1g-dev && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy requirements and install dependencies
COPY requirements.txt .
RUN pip install --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy application files
COPY . .

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONPATH=/app \
    PORT=4006 \
    HOST=0.0.0.0 \
    RAGNOR_DEBUG_IMAGES_PATH=/app/debug_images

# Create debug images directory if environment variable is set
RUN mkdir -p "$RAGNOR_DEBUG_IMAGES_PATH"

# Expose port
EXPOSE 3002

# Command to run the application
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "3002"]

