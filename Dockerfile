FROM debian:trixie-slim

ENV DEBIAN_FRONTEND=noninteractive

# System packages
RUN apt update && apt install -y --no-install-recommends \
        # Core utilities
        ca-certificates \
        curl \
        unzip \
        # Git
        git \
        # GUI / Rendering libraries
        libgl1 \
        libgtk2.0-dev \
        tk \
    && rm -rf /var/lib/apt/lists/*

# uv installation
RUN curl -LsSf https://astral.sh/uv/install.sh | sh

# nvm installation
ENV NVM_DIR=/root/.nvm
RUN curl -o- https://raw.githubusercontent.com/nvm-sh/nvm/v0.40.1/install.sh | bash \
    && . "$NVM_DIR/nvm.sh"