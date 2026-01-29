FROM apache/airflow:3.0.3-python3.12

USER root

# Install system dependencies for PDF processing
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        build-essential \
        curl \
        libpq-dev \
        poppler-utils \
        tesseract-ocr \
        libgl1-mesa-glx \
        libglib2.0-0 \
        && rm -rf /var/lib/apt/lists/*

USER airflow

# Copy and install Python dependencies
COPY requirements-airflow.txt /tmp/requirements-airflow.txt
RUN pip install --no-cache-dir -r /tmp/requirements-airflow.txt

# Copy entrypoint script
COPY --chmod=755 entrypoint.sh /entrypoint.sh

WORKDIR /opt/airflow

EXPOSE 8080

ENTRYPOINT ["/entrypoint.sh"]
