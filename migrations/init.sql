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
  status TEXT NOT NULL DEFAULT 'pendiente',
  created_at TIMESTAMPTZ DEFAULT NOW(),
  embedding vector(1536)
);

-- Indexes for fast geo + vector search
CREATE INDEX IF NOT EXISTS idx_reports_geo
  ON reports USING GIST (CAST(ST_MakePoint(longitude, latitude) AS geography));

-- Vector index (cosine distance)
CREATE INDEX IF NOT EXISTS idx_reports_embedding
  ON reports USING ivfflat (embedding vector_cosine_ops) WITH (lists = 100);
