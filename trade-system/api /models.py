# Pydantic models = input/output shapes for the API endpoints.
# They help validation and auto-generate Swagger docs.

from pydantic import BaseModel
from typing import List, Optional, Dict, Any

class ParseIn(BaseModel):
    # Plain English text from the user
    text: str

class ParsedPayload(BaseModel):
    # What we return after parsing the text
    origin: Optional[str] = None
    destination: Optional[str] = None
    hs: Optional[str] = None
    value: Optional[float] = None
    packaging: Optional[str] = "standard"
    flags: List[str] = []

class CompileIn(BaseModel):
    # Minimal info needed to run the rules engine
    origin: str
    destination: str
    hs: str
    value: Optional[float] = 0
    packaging: str = "standard"
    flags: List[str] = []

class CompileOut(BaseModel):
    # What the rules engine returns
    required_forms: List[str]
    rationale: List[str]
    # Optional: include the fields for those forms (for the UI to render)
    fields: List[Dict[str, Any]] = []

class SearchIn(BaseModel):
    # Query to search our harvested documents
    q: str
    k: int = 5

