#!/usr/bin/env bash
set -euo pipefail

APP_DIR="${APP_DIR:-/opt/agenteia}"
REPO_URL="${REPO_URL:-https://github.com/Luiisitaz/agenteia.git}"
REPO_DIR="${REPO_DIR:-$APP_DIR/agenteia}"
ENV_FILE="${ENV_FILE:-$REPO_DIR/.env}"

echo "[deploy] Installing Docker + docker-compose + git"
sudo apt-get update -y
sudo apt-get install -y docker.io docker-compose git
sudo systemctl enable --now docker

echo "[deploy] Preparing directory: $APP_DIR"
sudo mkdir -p "$APP_DIR"
sudo chown -R "$USER":"$USER" "$APP_DIR"

if [ ! -d "$REPO_DIR/.git" ]; then
  echo "[deploy] Cloning repo: $REPO_URL"
  git clone "$REPO_URL" "$REPO_DIR"
else
  echo "[deploy] Repo exists, pulling latest changes"
  (cd "$REPO_DIR" && git pull)
fi

if [ ! -f "$ENV_FILE" ]; then
  if [ -n "${OPENAI_API_KEY:-}" ] && [ -n "${GOOGLE_MAPS_API_KEY:-}" ]; then
    echo "[deploy] Creating .env from environment variables"
    cat > "$ENV_FILE" <<EOF
OPENAI_API_KEY=${OPENAI_API_KEY}
OPENAI_BASE_URL=${OPENAI_BASE_URL:-https://api.openai.com/v1}
GOOGLE_MAPS_API_KEY=${GOOGLE_MAPS_API_KEY}
LOG_LEVEL=${LOG_LEVEL:-info}
EOF
  else
    echo "[deploy] .env not found. Creating template at $ENV_FILE"
    cat > "$ENV_FILE" <<'EOF'
OPENAI_API_KEY=sk-...
OPENAI_BASE_URL=https://api.openai.com/v1
GOOGLE_MAPS_API_KEY=AIza...
LOG_LEVEL=info
EOF
    echo "[deploy] Edit $ENV_FILE with real keys and re-run the script."
    exit 1
  fi
fi

echo "[deploy] Starting containers"
cd "$REPO_DIR"
sudo docker-compose up -d --build
sudo docker-compose ps
echo "[deploy] Done. App should be on http://<SERVER_IP>:8000/app"
