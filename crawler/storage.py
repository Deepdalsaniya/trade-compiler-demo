# crawler/storage.py
import sqlite3, pathlib, time

SCHEMA = """
CREATE TABLE IF NOT EXISTS documents(
  id INTEGER PRIMARY KEY,
  source TEXT, url TEXT UNIQUE, fetched_at INTEGER,
  status INTEGER, http_status INTEGER,
  content_type TEXT, sha256 TEXT,
  title TEXT, text_excerpt TEXT
);
CREATE TABLE IF NOT EXISTS extractions(
  id INTEGER PRIMARY KEY,
  url TEXT, extracted_at INTEGER,
  kind TEXT,   -- 'forms','hs','thresholds','prohibitions'
  payload TEXT -- JSON
);
CREATE INDEX IF NOT EXISTS idx_docs_url ON documents(url);
"""

def get_db(path="crawler.db"):
  con = sqlite3.connect(path)
  con.execute("PRAGMA journal_mode=WAL")
  con.executescript(SCHEMA)
  return con

def upsert_doc(con, row):
  cols = ",".join(row.keys())
  qmarks = ",".join([":" + k for k in row.keys()])
  con.execute(f"INSERT OR REPLACE INTO documents({cols}) VALUES({qmarks})", row)
  con.commit()

