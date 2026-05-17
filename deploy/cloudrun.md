# Deploying Gap2Idea to Google Cloud Run

Streamlit app → container → Cloud Run, with secrets in Secret Manager.

## Prerequisites

- `gcloud` CLI installed and authenticated (`gcloud auth login`).
- A GCP project with billing enabled.
- Your `OPENROUTER_API_KEY` (required) and `S2_API_KEY` (optional) ready.

## Variables (set once per shell)

PowerShell:

```powershell
$env:PROJECT_ID = "your-gcp-project-id"
$env:REGION     = "europe-west1"           # or us-central1, etc.
$env:SERVICE    = "gap2idea"
$env:REPO       = "gap2idea"               # Artifact Registry repo name
$env:IMAGE      = "$env:REGION-docker.pkg.dev/$env:PROJECT_ID/$env:REPO/$env:SERVICE`:latest"
gcloud config set project $env:PROJECT_ID
```

Bash:

```bash
export PROJECT_ID=your-gcp-project-id
export REGION=europe-west1
export SERVICE=gap2idea
export REPO=gap2idea
export IMAGE="$REGION-docker.pkg.dev/$PROJECT_ID/$REPO/$SERVICE:latest"
gcloud config set project "$PROJECT_ID"
```

## 1. Enable APIs (once per project)

```bash
gcloud services enable \
  run.googleapis.com \
  artifactregistry.googleapis.com \
  cloudbuild.googleapis.com \
  secretmanager.googleapis.com
```

## 2. Create the Artifact Registry repo (once)

```bash
gcloud artifacts repositories create "$REPO" \
  --repository-format=docker \
  --location="$REGION" \
  --description="Gap2Idea container images"
```

## 3. Store secrets in Secret Manager (once; rotate as needed)

```bash
# Required
printf "%s" "sk-or-v1-..." | gcloud secrets create OPENROUTER_API_KEY --data-file=-

# Optional
printf "%s" "your-s2-key" | gcloud secrets create S2_API_KEY --data-file=-
```

To update an existing secret, add a new version:

```bash
printf "%s" "new-value" | gcloud secrets versions add OPENROUTER_API_KEY --data-file=-
```

## 4. Build the image with Cloud Build

From the repo root (the directory with `Dockerfile`):

```bash
gcloud builds submit --tag "$IMAGE" .
```

(`gcloud builds submit` honours `.gcloudignore`, so `.venv/`, `data/`, `artifacts/` won't be uploaded.)

## 5. Deploy to Cloud Run

```bash
gcloud run deploy "$SERVICE" \
  --image "$IMAGE" \
  --region "$REGION" \
  --platform managed \
  --allow-unauthenticated \
  --port 8080 \
  --memory 4Gi \
  --cpu 2 \
  --min-instances 0 \
  --max-instances 3 \
  --timeout 3600 \
  --session-affinity \
  --set-secrets "OPENROUTER_API_KEY=OPENROUTER_API_KEY:latest,S2_API_KEY=S2_API_KEY:latest"
```

Key flags:

- `--memory 4Gi` — sentence-transformers + the embedding model need headroom; 2Gi is too tight.
- `--session-affinity` — Streamlit uses websockets; affinity keeps each user on one instance.
- `--timeout 3600` — long-running idea-generation requests won't be killed mid-flight.
- `--allow-unauthenticated` — public URL. Drop this if you want IAM-gated access.

The command prints a `https://gap2idea-<hash>-<region>.a.run.app` URL when it finishes.

## 6. Subsequent deploys

After code changes:

```bash
gcloud builds submit --tag "$IMAGE" .
gcloud run deploy "$SERVICE" --image "$IMAGE" --region "$REGION"
```

(The second command keeps all previously-set flags; you only need them again to change them.)

## Notes & gotchas

- **Cold starts**: first request after idle scales an instance up; loading the embedding model on boot takes 10–30 s. Set `--min-instances 1` (costs ~$30/mo) if you want instant response.
- **Filesystem is ephemeral**: anything written to `data/` or `artifacts/` inside the container vanishes when the instance shuts down. For persistent outputs, mount a GCS bucket with the Cloud Run GCS volume (gen2) feature or write to a Cloud SQL / Firestore.
- **Cost ceiling**: with `--max-instances 3` and scale-to-zero, idle cost is ~$0. Each request running 4Gi/2vCPU bills ~$0.00009/sec.
- **CPU during requests only**: the default `--cpu-throttling` is fine for an interactive UI. If you run the batch pipeline inside the container, add `--no-cpu-throttling` (charged for the full instance lifetime).
- **Auth-gated deploy**: drop `--allow-unauthenticated`, then grant `roles/run.invoker` to specific users / groups.

## Tearing it all down

```bash
gcloud run services delete "$SERVICE" --region "$REGION"
gcloud artifacts repositories delete "$REPO" --location "$REGION"
gcloud secrets delete OPENROUTER_API_KEY
gcloud secrets delete S2_API_KEY
```
