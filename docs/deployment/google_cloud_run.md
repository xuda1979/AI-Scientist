# Deploying AI Scientist to Google Cloud Run

This guide walks through packaging the AI Scientist workflow into a container
image and deploying it as a managed web service on [Google Cloud Run](https://cloud.google.com/run).
The deployment exposes a REST API that lets you trigger workflow executions and
poll their status from any HTTP client.

## Overview

1. **Container image** – `deploy/cloud_run/Dockerfile` builds a container that
   bundles the AI Scientist source code, installs all Python dependencies (plus
   FastAPI for the web API), and exposes the service using Uvicorn.
2. **Deployment script** – `deploy/cloud_run/deploy_to_cloud_run.sh` wraps the
   Cloud Build and Cloud Run commands needed to publish the service.
3. **REST API** – `deploy/cloud_run/app.py` defines endpoints for creating new
   workflow jobs, checking job status, and a `/healthz` probe for uptime checks.
4. **Optional archiving** – Completed runs can be automatically compressed and
   uploaded to Cloud Storage when a bucket name is provided.

## Prerequisites

Before you begin, make sure you have:

- A Google Cloud project with billing enabled.
- The [Google Cloud SDK](https://cloud.google.com/sdk/docs/install) installed
  and authenticated (`gcloud auth login`).
- The `gcloud` CLI configured to use your project (`gcloud config set project <PROJECT_ID>`).
- (Optional) A Google Cloud Storage bucket if you plan to archive workflow
  artifacts.

## 1. Configure environment variables

The deployment script reads configuration from environment variables. Set the
required values in your shell (replace the placeholders with your own values):

```bash
export PROJECT_ID="my-gcp-project"
export SERVICE_NAME="ai-scientist-service"

# Optional overrides
export REGION="us-central1"          # Deploy region
export MAX_CONCURRENT_JOBS=1         # Parallel workflow executions per instance
export OUTPUT_BASE_DIR="/workspace/output"  # Where to write artifacts inside the container
export RESULTS_GCS_BUCKET="my-artifacts"    # (Optional) Bucket for archived results
export RESULTS_GCS_PREFIX="ai-scientist"    # (Optional) Folder prefix inside the bucket
export ALLOW_UNAUTH=true             # Allow public HTTP access
export OPENAI_SECRET_NAME="openai-api-key"  # (Optional) Secret Manager name for OpenAI key
export GENAI_SECRET_NAME="google-api-key"   # (Optional) Secret Manager name for Gemini key
```

> **Tip:** If you need to run behind a VPC connector or custom service account,
> set `VPC_CONNECTOR` or `SERVICE_ACCOUNT` before running the script.

## 2. Provision API keys and secrets

The workflow requires API keys for the language models you plan to use. Create
secrets in Secret Manager so they can be mounted securely by Cloud Run:

```bash
# OpenAI key (replace with your value)
echo -n "sk-..." | gcloud secrets create openai-api-key --replication-policy="automatic" --data-file=-

# Google Generative AI key (optional)
echo -n "AIza..." | gcloud secrets create google-api-key --replication-policy="automatic" --data-file=-
```

The deployment script automatically wires these secrets to environment
variables inside the Cloud Run service when `OPENAI_SECRET_NAME` or
`GENAI_SECRET_NAME` are set.

## 3. Run the deployment script

Make the script executable and launch the deployment:

```bash
chmod +x deploy/cloud_run/deploy_to_cloud_run.sh
./deploy/cloud_run/deploy_to_cloud_run.sh
```

The script performs the following actions:

1. Enables the Cloud Run, Artifact Registry, and Cloud Build APIs.
2. Builds the container image using `deploy/cloud_run/Dockerfile`.
3. Pushes the image to the registry configured by `IMAGE_REPO`.
4. Deploys (or updates) the Cloud Run service with your configuration.

Take note of the service URL displayed at the end—it will be used to invoke the
API. If you disabled unauthenticated access you must authenticate using an
Identity Token (`gcloud auth print-identity-token`).

## 4. Grant bucket permissions (optional)

If you configured `RESULTS_GCS_BUCKET`, grant the Cloud Run runtime service
account permission to write to the bucket:

```bash
RUNTIME_SA=$(gcloud run services describe "$SERVICE_NAME" \
  --project "$PROJECT_ID" \
  --region "$REGION" \
  --format 'value(spec.template.spec.serviceAccountName)')

gcloud storage buckets add-iam-policy-binding "gs://$RESULTS_GCS_BUCKET" \
  --member "serviceAccount:$RUNTIME_SA" \
  --role roles/storage.objectCreator
```

## 5. Use the REST API

### Create a workflow job

```bash
SERVICE_URL=$(gcloud run services describe "$SERVICE_NAME" \
  --project "$PROJECT_ID" \
  --region "$REGION" \
  --format 'value(status.url)')

curl -X POST "$SERVICE_URL/jobs" \
  -H 'Content-Type: application/json' \
  -d '{
        "topic": "AI Planning",
        "field": "Computer Science",
        "question": "How can LLMs plan more efficiently?",
        "max_iterations": 2,
        "enable_ideation": false
      }'
```

The response includes a `job_id` you can poll for status updates.

### Check job status

```bash
curl "$SERVICE_URL/jobs/<job_id>"
```

Once a job finishes you will see the `result_dir` path inside the container. If
Cloud Storage archiving is enabled the response also contains an `artifact_uri`
with a downloadable ZIP file.

### Health checks

Cloud Run automatically sends HTTP health checks to `/healthz`. You can hit the
same endpoint manually to confirm the service is running:

```bash
curl "$SERVICE_URL/healthz"
```

## 6. Cleaning up

To remove all deployed resources:

```bash
gcloud run services delete "$SERVICE_NAME" \
  --project "$PROJECT_ID" \
  --region "$REGION"
```

Delete container images from Artifact Registry/Container Registry if desired:

```bash
gcloud container images delete "$IMAGE_URI" --force-delete-tags
```

Finally, remove the secrets or buckets you no longer need.

## Troubleshooting

- **Image build fails:** Ensure all Python dependencies install correctly. You
  may need to increase Cloud Build timeout for large TeX packages
  (`gcloud builds submit --timeout=30m ...`).
- **Permission denied on bucket:** Double-check that the runtime service account
  has `roles/storage.objectCreator` on the target bucket.
- **API returns 500 errors:** View the service logs with `gcloud run services
  logs tail $SERVICE_NAME --region $REGION` to inspect stack traces.
- **Long-running workflows:** Increase the Cloud Run request timeout or consider
  moving heavy workloads to Cloud Run Jobs. You can reuse the container image
  with `gcloud run jobs create` if needed.

With these steps you can operate AI Scientist as a web-accessible research
assistant backed by Google Cloud's managed infrastructure.
