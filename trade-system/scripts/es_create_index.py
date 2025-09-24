# Create an Elasticsearch index to store our docs for keyword search.

from elasticsearch import Elasticsearch

es = Elasticsearch(hosts=["http://localhost:9200"])
idx = "compliance_docs"

mapping = {
  "settings": {
    "analysis": {
      "analyzer": {
        "default": { "type":"standard" }
      }
    }
  },
  "mappings": {
    "properties": {
      "url":   {"type":"keyword"},
      "title": {"type":"text"},
      "text":  {"type":"text"}
    }
  }
}

if es.indices.exists(index=idx):
    print("Index already exists")
else:
    es.indices.create(index=idx, body=mapping)
    print("Created index:", idx)

