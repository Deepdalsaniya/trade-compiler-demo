# The brain that decides which forms are required based on inputs and DB rules.

from typing import Dict, List
from api.db import fetchall

def hs_prefixes(hs: str) -> List[str]:
    """
    Turn '8517.62' into a set of prefixes that rules can match:
    ['8517', '8517.62', '851762', '8517.6', '851762?'] (we keep a few variants)
    We keep it simple but helpful for matching chapter/heading-level rules.
    """
    hs = (hs or "").replace(" ", "")
    parts = [p for p in hs.replace("-", ".").split(".") if p]
    out = []
    if len(parts) >= 1: out.append(parts[0])               # '8517'
    if len(parts) >= 2:
        out.append(parts[0] + parts[1])                    # '851762' (glued)
        out.append(parts[0] + "." + parts[1])              # '8517.62' (dotted)
    if len(parts) >= 3:
        out += [
            parts[0] + parts[1] + parts[2],               # '851762xx' style
            parts[0] + "." + parts[1] + "." + parts[2],   # '8517.62.xx'
            parts[0] + parts[1] + "." + parts[2],         # '851762.xx'
        ]
    # De-dupe while keeping order
    seen, res = set(), []
    for p in out:
        if p not in seen:
            res.append(p); seen.add(p)
    return res

def cond_true(cond: Dict, payload: Dict) -> bool:
    """
    Check if a single rule's condition matches the current shipment payload.
    We support only a few keys for now; you can add more later.
    """
    if "origin" in cond and payload.get("origin") != cond["origin"]: return False
    if "destination_in" in cond and payload.get("destination") not in cond["destination_in"]: return False
    if "hs_prefix_in" in cond:
        prefs = payload.get("hs_prefixes", [])
        # True if ANY target matches ANY of the prefixes
        if not any(any(p.startswith(t) or p == t for p in prefs) for t in cond["hs_prefix_in"]):
            return False
    if "value_gt" in cond:
        v = payload.get("value")
        if v is None or float(v) <= float(cond["value_gt"]): return False
    if "packaging_eq" in cond and payload.get("packaging") != cond["packaging_eq"]: return False
    if "flags_has" in cond and cond["flags_has"] not in payload.get("flags", []): return False
    return True

def compile_required(payload: Dict) -> (List[str], List[str]):
    """
    Run all active rules (ordered by priority) and collect the forms they add.
    We always include CI + PL as the base set.
    """
    rules = fetchall("""
        SELECT name, priority, condition_json, add_forms
        FROM rules
        WHERE active = true
        ORDER BY priority ASC
    """)
    required = {"CI", "PL"}
    rationale = ["Core: CI, PL"]
    payload["hs_prefixes"] = hs_prefixes(payload.get("hs",""))

    for r in rules:
        if cond_true(r["condition_json"], payload):
            for f in r["add_forms"]:
                required.add(f)
            rationale.append(f"Rule matched: {r['name']}")

    return sorted(required), rationale

def fields_for(forms: List[str]) -> List[Dict]:
    """
    Return the field definitions for the forms we need.
    The UI can render this to collect values from the user.
    """
    if not forms:
        return []
    q = """
      SELECT form_code, field_key, label, type, required
      FROM form_fields
      WHERE form_code = ANY(%s)
      ORDER BY form_code, field_key
    """
    rows = fetchall(q, (forms,))
    return rows

