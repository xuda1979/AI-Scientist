#!/usr/bin/env bash
#
# Deploy the AI Scientist Cloud Run service.
#
# This script builds the container image, pushes it to Artifact Registry (or
# Container Registry), and deploys the Cloud Run service with sensible defaults.
# Configuration is handled via environment variables so the script can be
# customised without editing the file.  Run with `bash -e` to fail fast.

set -euo pipefail

# Verify gcloud is installed before continuing.
if ! command -v gcloud >/dev/null 2>&1; then
  echo "gcloud CLI not found. Install the Google Cloud SDK before running this script." >&2
  exit 1
fi

# Required configuration ----------------------------------------------------
: "${PROJECT_ID:?Set PROJECT_ID to your Google Cloud project ID}"
: "${SERVICE_NAME:?Set SERVICE_NAME to the desired Cloud Run service name}"

# Optional configuration with defaults --------------------------------------
REGION=${REGION:-us-central1}
PORT=${PORT:-8080}
IMAGE_REPO=${IMAGE_REPO:-gcr.io}
IMAGE_TAG=${IMAGE_TAG:-$(date +%Y%m%d-%H%M%S)}
IMAGE_URI=${IMAGE_URI:-${IMAGE_REPO}/${PROJECT_ID}/${SERVICE_NAME}:${IMAGE_TAG}}
MAX_CONCURRENT_JOBS=${MAX_CONCURRENT_JOBS:-1}
OUTPUT_BASE_DIR=${OUTPUT_BASE_DIR:-/workspace/output}
RESULTS_GCS_BUCKET=${RESULTS_GCS_BUCKET:-}
RESULTS_GCS_PREFIX=${RESULTS_GCS_PREFIX:-ai-scientist}
ALLOW_UNAUTH=${ALLOW_UNAUTH:-true}
SERVICE_ACCOUNT=${SERVICE_ACCOUNT:-}
VPC_CONNECTOR=${VPC_CONNECTOR:-}
CPU=${CPU:-2}
MEMORY=${MEMORY:-4Gi}
CONCURRENCY=${CONCURRENCY:-1}
MIN_INSTANCES=${MIN_INSTANCES:-0}
MAX_INSTANCES=${MAX_INSTANCES:-1}
OPENAI_SECRET_NAME=${OPENAI_SECRET_NAME:-}
GENAI_SECRET_NAME=${GENAI_SECRET_NAME:-}

# Informational output so users can verify configuration.
echo "Project:           ${PROJECT_ID}"
echo "Service:           ${SERVICE_NAME}"
echo "Region:            ${REGION}"
echo "Image URI:         ${IMAGE_URI}"
echo "Output directory:  ${OUTPUT_BASE_DIR}"
echo "Max jobs:          ${MAX_CONCURRENT_JOBS}"
if [[ -n "${RESULTS_GCS_BUCKET}" ]]; then
  echo "Artifact bucket:   gs://${RESULTS_GCS_BUCKET}/${RESULTS_GCS_PREFIX}"
fi

# Ensure required services are enabled. These commands are idempotent.
gcloud services enable \
  run.googleapis.com \
  artifactregistry.googleapis.com \
  cloudbuild.googleapis.com \
  --project "${PROJECT_ID}"

# Build and push the container image using Cloud Build.
gcloud builds submit \
  --project "${PROJECT_ID}" \
  --tag "${IMAGE_URI}" \
  --file deploy/cloud_run/Dockerfile \
  .

# Construct environment variable arguments for Cloud Run deployment.
ENV_VARS=(
  "OUTPUT_BASE_DIR=${OUTPUT_BASE_DIR}"
  "MAX_CONCURRENT_JOBS=${MAX_CONCURRENT_JOBS}"
)
if [[ -n "${RESULTS_GCS_BUCKET}" ]]; then
  ENV_VARS+=("RESULTS_GCS_BUCKET=${RESULTS_GCS_BUCKET}")
fi
if [[ -n "${RESULTS_GCS_PREFIX}" ]]; then
  ENV_VARS+=("RESULTS_GCS_PREFIX=${RESULTS_GCS_PREFIX}")
fi

ENV_VARS_ARG=$(IFS=, ; echo "${ENV_VARS[*]}")

# Assemble optional deployment flags.
read -r -a EXTRA_FLAGS <<< ""
if [[ "${ALLOW_UNAUTH}" == "true" ]]; then
  EXTRA_FLAGS+=("--allow-unauthenticated")
fi
if [[ -n "${SERVICE_ACCOUNT}" ]]; then
  EXTRA_FLAGS+=("--service-account" "${SERVICE_ACCOUNT}")
fi
if [[ -n "${VPC_CONNECTOR}" ]]; then
  EXTRA_FLAGS+=("--vpc-connector" "${VPC_CONNECTOR}")
  EXTRA_FLAGS+=("--egress" "all")
fi
if [[ -n "${OPENAI_SECRET_NAME}" ]]; then
  EXTRA_FLAGS+=("--set-secrets" "OPENAI_API_KEY=${OPENAI_SECRET_NAME}:latest")
fi
if [[ -n "${GENAI_SECRET_NAME}" ]]; then
  EXTRA_FLAGS+=("--set-secrets" "GOOGLE_API_KEY=${GENAI_SECRET_NAME}:latest")
fi

# Deploy to Cloud Run.
gcloud run deploy "${SERVICE_NAME}" \
  --project "${PROJECT_ID}" \
  --image "${IMAGE_URI}" \
  --platform managed \
  --region "${REGION}" \
  --port "${PORT}" \
  --cpu "${CPU}" \
  --memory "${MEMORY}" \
  --concurrency "${CONCURRENCY}" \
  --min-instances "${MIN_INSTANCES}" \
  --max-instances "${MAX_INSTANCES}" \
  --set-env-vars "${ENV_VARS_ARG}" \
  "${EXTRA_FLAGS[@]}"

cat <<'EOF'
Deployment complete!

Next steps:
  • Note the service URL printed above.
  • Grant the Cloud Run runtime service account access to any Cloud Storage
    bucket used for result archiving.
  • Use `curl` or any HTTP client to create workflow jobs:
        curl -X POST "$SERVICE_URL/jobs" \
             -H 'Content-Type: application/json' \
             -d '{"topic":"AI Planning","field":"Computer Science","question":"How can LLMs plan more efficiently?"}'
EOF
