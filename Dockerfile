FROM python:3.13-slim

ARG TARGETPLATFORM=linux/amd64
ARG DEBIAN_FRONTEND=noninteractive
ARG LANG=C.UTF-8

# Install system dependencies
RUN apt-get update && apt-get install -y \
    wget \
    netcat-traditional \
    gnupg \
    curl \
    unzip \
    supervisor \
    net-tools \
    procps \
    git \
    && rm -rf /var/lib/apt/lists/*

# Set platform for ARM64 compatibility
ENV OPENAI_BASE_URL="https://api.openai.com/v1"
ENV OPENAI_API_KEY="sk-XXX"

# Set up working directory
WORKDIR /app

# Copy requirements and install backend dependencies
COPY scripts/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Set up supervisor configuration
RUN mkdir -p /var/log/supervisor
COPY scripts/supervisord.conf /etc/supervisor/conf.d/supervisord.conf
COPY scripts/entrypoint.sh /entrypoint.sh

EXPOSE 8000
RUN chmod +x /entrypoint.sh
ENTRYPOINT ["/entrypoint.sh"]
