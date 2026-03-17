import os
import sys
from pathlib import Path
from sqlalchemy import create_engine, text
from sqlalchemy.engine import make_url
from dotenv import load_dotenv
import configparser

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.reporting import priority_case_sql
load_dotenv(ROOT / ".env")

CONFIG = configparser.ConfigParser()
CONFIG.read(ROOT / "config.ini")

DEFAULT_DB_URL = "postgresql+psycopg2://postgres:12345@localhost:5433/urban_reports"
DATABASE_URL = os.getenv("DATABASE_URL", DEFAULT_DB_URL)

EMBED_DIM = int(CONFIG.get("openai", "embedding_dimensions", fallback="1536"))
PRIORITY_CASE = priority_case_sql("report_type")

SCHEMA_SQL = f"""
CREATE EXTENSION IF NOT EXISTS postgis;
CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE IF NOT EXISTS reports (
  id SERIAL PRIMARY KEY,
  cedula VARCHAR(20) NOT NULL,
  report_type TEXT NOT NULL,
  description TEXT NOT NULL,
  location_text TEXT,
  latitude DOUBLE PRECISION NOT NULL,
  longitude DOUBLE PRECISION NOT NULL,
  priority TEXT NOT NULL DEFAULT 'medium',
  status TEXT NOT NULL DEFAULT 'pendiente',
  created_at TIMESTAMPTZ DEFAULT NOW(),
  embedding vector({EMBED_DIM})
);

ALTER TABLE reports
  ADD COLUMN IF NOT EXISTS priority TEXT;

ALTER TABLE reports
  ALTER COLUMN priority SET DEFAULT 'medium';

UPDATE reports
SET priority = {PRIORITY_CASE}
WHERE priority IS NULL OR priority = '';

ALTER TABLE reports
  ALTER COLUMN priority SET NOT NULL;

CREATE INDEX IF NOT EXISTS idx_reports_geo
  ON reports USING GIST (CAST(ST_MakePoint(longitude, latitude) AS geography));

CREATE INDEX IF NOT EXISTS idx_reports_embedding
  ON reports USING ivfflat (embedding vector_cosine_ops) WITH (lists = 100);
"""

def ensure_database(url_str: str) -> None:
    url = make_url(url_str)
    db_name = url.database
    admin_url = url.set(database="postgres")
    engine = create_engine(admin_url)
    with engine.connect().execution_options(isolation_level="AUTOCOMMIT") as conn:
        exists = conn.execute(text("SELECT 1 FROM pg_database WHERE datname = :name"), {"name": db_name}).scalar()
        if not exists:
            conn.execute(text(f'CREATE DATABASE "{db_name}"'))


def apply_schema(url_str: str) -> None:
    engine = create_engine(url_str)
    with engine.connect().execution_options(isolation_level="AUTOCOMMIT") as conn:
        statements = [stmt.strip() for stmt in SCHEMA_SQL.split(";") if stmt.strip()]
        for stmt in statements:
            conn.exec_driver_sql(stmt)


if __name__ == "__main__":
    print("[init_db] Using database:", DATABASE_URL)
    ensure_database(DATABASE_URL)
    apply_schema(DATABASE_URL)
    print("[init_db] Done.")
