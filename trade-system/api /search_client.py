# Simple helper to query Elasticsearch for relevant documents.
# Use this to show users the official references we indexed.

from elasticsearch import Elasticsearch
import os

def get_es():
    host = os.getenv("ES_HOST","http://localhost:9200")
    return Elasticsearch(hosts=[host])

def search(q: str, k: int = 5):
    es = get_es()
    idx = "compliance_docs"
    # If index doesn't exist yet, return empty results
    if not es.indices.exists(index=idx):
        return []
    # Basic multi-field search: title is 3x more important than body text
    body = {
        "query": {"multi_match": {"query": q, "fields": ["title^3","text"]}},
        "size": k
    }
    res = es.search(index=idx, body=body)
    hits = res.get("hits", {}).get("hits", [])
    return [{
        "url": h["_source"]["url"],
        "title": h["_source"].get("title"),
        "score": h["_score"]
    } for h in hits]

