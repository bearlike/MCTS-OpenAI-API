FROM python:3.13-slim

# Install system dependencies
RUN apt-get update && apt-get install -y \
    wget \
    netcat-traditional \
    gnupg \
    curl \
    unzip \
    xvfb \
    libgconf-2-4 \
    libxss1 \
    libnss3 \
    libnspr4 \
    libasound2 \
    libatk1.0-0 \
    libatk-bridge2.0-0 \
    libcups2 \
    libdbus-1-3 \
    libdrm2 \
    libgbm1 \
    libgtk-3-0 \
    libxcomposite1 \
    libxdamage1 \
    libxfixes3 \
    libxrandr2 \
    xdg-utils \
    fonts-liberation \
    dbus \
    xauth \
    xvfb \
    x11vnc \
    tigervnc-tools \
    supervisor \
    net-tools \
    procps \
    git \
    python3-numpy \
    fontconfig \
    fonts-dejavu \
    fonts-dejavu-core \
    fonts-dejavu-extra

# Set platform for ARM64 compatibility
ARG TARGETPLATFORM=linux/amd64
ENV OPENAI_BASE_URL="https://api.openai.com/v1"
ENV OPENAI_API_KEY="sk-XXX"

# Set up working directory
WORKDIR /app

# Copy requirements and install backend dependencies
COPY scripts/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .
RUN rm -rf /var/lib/apt/lists/*

# Set up supervisor configuration
RUN mkdir -p /var/log/supervisor
COPY scripts/supervisord.conf /etc/supervisor/conf.d/supervisord.conf
COPY scripts/entrypoint.sh /entrypoint.sh

EXPOSE 8000
RUN chmod +x /entrypoint.sh

LABEL org.opencontainers.image.source https://github.com/bearlike/mcts-openai-api
ENTRYPOINT ["/entrypoint.sh"]
