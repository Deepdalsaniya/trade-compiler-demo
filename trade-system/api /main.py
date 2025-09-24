# FastAPI = our backend service.
# Endpoints:
#  - /health   -> quick health check
#  - /parse    -> (placeholder) turn English into structured data (we’ll improve later)
#  - /compile  -> run rules engine and return required forms + fields
#  - /search   -> keyword search across harvested docs

from fastapi import FastAPI
from api.models import ParseIn, ParsedPayload, CompileIn, CompileOut, SearchIn
from api.rules_engine import compile_required, fields_for
from api.search_client import search

app = FastAPI(title="Trade Compliance API")

@app.get("/health")
def health():
    # Simple ping endpoint: return {"ok": true} if server is up
    return {"ok": True}

@app.post("/parse", response_model=ParsedPayload)
def parse(inp: ParseIn):
    """
    Very simple parser for now (demo):
    - If text contains certain words, we guess origin/destination/hs.
    Replace later with your real LLM+heuristics function.
    """
    t = inp.text.lower()
    origin = "US" if "us" in t or "united states" in t else ("IN" if "india" in t else None)
    destination = "DE" if "germany" in t else ("LK" if "lanka" in t or "sri lanka" in t else None)
    hs = "85" if "electronics" in t or "router" in t else ("0409" if "honey" in t else None)
    packaging = "wood" if ("pallet" in t or "wood" in t) else "standard"
    return ParsedPayload(origin=origin, destination=destination, hs=hs, packaging=packaging, flags=[])

@app.post("/compile", response_model=CompileOut)
def compile(inp: CompileIn):
    """
    Run the rules engine.
    Input: origin/destination/hs/value/packaging/flags
    Output: list of required forms, rationale, and field definitions for those forms.
    """
    payload = inp.dict()
    forms, rationale = compile_required(payload)
    frows = fields_for(forms)
    return CompileOut(required_forms=forms, rationale=rationale, fields=frows)

@app.post("/search")
def search_docs(s: SearchIn):
    """
    Search harvested documents (Elasticsearch BM25).
    Returns a few best matches with URL and score.
    """
    return {"results": search(s.q, s.k)}

