-- Turn on the vector extension (used later for semantic search)
CREATE EXTENSION IF NOT EXISTS vector;

-- Table of harvested documents (web pages / PDFs turned to text)
CREATE TABLE IF NOT EXISTS documents (
  id BIGSERIAL PRIMARY KEY,
  source TEXT,              -- where we got it (e.g., 'cbp', 'bis')
  url TEXT UNIQUE,          -- the page link
  title TEXT,               -- page title
  text TEXT,                -- extracted plain text
  sha256 TEXT,              -- hash to detect changes
  content_type TEXT,        -- e.g., 'text/html' or 'application/pdf'
  fetched_at TIMESTAMPTZ DEFAULT now(),
  meta JSONB DEFAULT '{}',  -- any extra metadata
  embedding vector(768)     -- placeholder for semantic embeddings (optional for later)
);

-- LLM-proposed rule candidates (we review these before making them active)
CREATE TABLE IF NOT EXISTS extractions (
  id BIGSERIAL PRIMARY KEY,
  url TEXT,                 -- which document this came from
  kind TEXT,                -- e.g., 'rules'
  payload JSONB,            -- the actual proposed rules as JSON
  confidence NUMERIC,       -- optional confidence score
  extracted_at TIMESTAMPTZ DEFAULT now(),
  status TEXT DEFAULT 'proposed' -- 'proposed' | 'approved' | 'rejected'
);

-- Production rules used by the engine
CREATE TABLE IF NOT EXISTS rules (
  id BIGSERIAL PRIMARY KEY,
  name TEXT,
  priority INT DEFAULT 100,          -- smaller = runs earlier
  active BOOLEAN DEFAULT TRUE,       -- switch rules on/off
  condition_json JSONB NOT NULL,     -- conditions to match (origin, value_gt, etc.)
  add_forms JSONB NOT NULL,          -- forms to add if conditions match
  version INT DEFAULT 1,
  source_url TEXT,                   -- where this rule came from
  created_at TIMESTAMPTZ DEFAULT now()
);

-- Form field definitions (which fields each form needs)
CREATE TABLE IF NOT EXISTS form_fields (
  id BIGSERIAL PRIMARY KEY,
  form_code TEXT,
  field_key TEXT,
  label TEXT,
  type TEXT,
  required BOOLEAN DEFAULT TRUE
);

-- Helpful indexes (speed up common queries)
CREATE INDEX IF NOT EXISTS idx_rules_active ON rules(active);
CREATE INDEX IF NOT EXISTS idx_rules_pri ON rules(priority);
CREATE INDEX IF NOT EXISTS idx_documents_url ON documents(url);

