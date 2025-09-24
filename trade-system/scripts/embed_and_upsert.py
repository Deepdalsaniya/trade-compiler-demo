# Pull documents from Postgres and index them into Elasticsearch.
# (Right now we only do BM25 keyword indexing; embeddings come later.)

import psycopg2, os
from psycopg2.extras import RealDictCursor
from elasticsearch import Elasticsearch

# Connect to Postgres
con = psycopg2.connect(
    host=os.getenv("PGHOST","localhost"),
    dbname=os.getenv("PGDATABASE","trade"),
    user=os.getenv("PGUSER","postgres"),
    password=os.getenv("PGPASSWORD","postgres")
)

# Connect to Elasticsearch
es = Elasticsearch(hosts=["http://localhost:9200"])
idx = "compliance_docs"

with con, con.cursor(cursor_factory=RealDictCursor) as cur:
    # Pull recent docs (limit keeps it quick for demos)
    cur.execute("SELECT id, url, title, text FROM documents ORDER BY id DESC LIMIT 500")
    for row in cur.fetchall():
        # Write each doc into ES
        es.index(index=idx, id=row["id"], document={
            "url": row["url"],
            "title": row["title"],
            "text": row["text"]
        })
        print("Indexed document:", row["id"], row["url"])

