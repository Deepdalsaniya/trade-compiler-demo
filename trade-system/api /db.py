# Simple helper to talk to Postgres.
# You call fetchall("SELECT ...", [params]) and get a list of dict rows.

import os
import psycopg2
from psycopg2.extras import RealDictCursor

def get_conn():
    # Connect using environment variables (set in .env / docker-compose)
    return psycopg2.connect(
        host=os.getenv("PGHOST","localhost"),
        port=os.getenv("PGPORT","5432"),
        user=os.getenv("PGUSER","postgres"),
        password=os.getenv("PGPASSWORD","postgres"),
        dbname=os.getenv("PGDATABASE","trade")
    )

def fetchall(query, params=None):
    # Run a query and return all rows as dicts
    with get_conn() as con, con.cursor(cursor_factory=RealDictCursor) as cur:
        cur.execute(query, params or [])
        return cur.fetchall()

