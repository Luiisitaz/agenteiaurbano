#!/usr/bin/env bash
set -euo pipefail

echo "[init_db] Starting database container..."
docker compose up -d db

echo "[init_db] Applying schema..."
python scripts/init_db.py
