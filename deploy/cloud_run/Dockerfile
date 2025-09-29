# syntax=docker/dockerfile:1
#
# Container image for deploying the AI Scientist workflow as a Cloud Run service.
# The image bundles the project source, installs the Python package along with
# the FastAPI server, and configures an entrypoint that serves the HTTP API.

FROM python:3.10-slim

# Install system dependencies required for LaTeX compilation and simulations.
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        build-essential \
        git \
        texlive-latex-extra \
        texlive-science \
        texlive-fonts-recommended \
        latexmk \
        wget \
    && rm -rf /var/lib/apt/lists/*

# Improve Python runtime behaviour in containers.
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    OUTPUT_BASE_DIR=/workspace/output

WORKDIR /app

# Copy the repository into the image.
COPY . /app

# Install the project along with the API server dependencies.
RUN pip install --upgrade pip && \
    pip install --no-cache-dir . fastapi uvicorn[standard] google-cloud-storage

# Create the default output directory used by the API service.
RUN mkdir -p ${OUTPUT_BASE_DIR}

# Expose the default HTTP port used by Cloud Run.
EXPOSE 8080

# Run the FastAPI application with uvicorn.
CMD ["uvicorn", "deploy.cloud_run.app:app", "--host", "0.0.0.0", "--port", "8080"]
