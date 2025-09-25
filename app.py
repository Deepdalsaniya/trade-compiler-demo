# app.py — Compliance Compiler (AI-first, HS normalized, no irrelevant forms)

import json, re
import pandas as pd
import streamlit as st

st.set_page_config(page_title="Compliance Compiler", page_icon="🌍")

# -------------------- CONFIG --------------------
RULES_URL = st.secrets.get("RULES_CSV_URL", "rules.csv")
FIELDS_URL = st.secrets.get("FIELDS_CSV_URL", "fields.csv")  # kept for future (not rendered)

# Optional PDF support
try:
    from pdf_utils import make_ci_pdf, make_pl_pdf
    PDF_OK = True
except Exception:
    PDF_OK = False

# -------------------- LLM (optional) --------------------
try:
    from openai import OpenAI
    _OPENAI = OpenAI(api_key=st.secrets.get("OPENAI_API_KEY")) if st.secrets.get("OPENAI_API_KEY") else None
except Exception:
    _OPENAI = None

# -------------------- Dictionaries / Hints --------------------
_COUNTRY_MAP = {
    "united states":"US","usa":"US","us":"US","america":"US","u.s.":"US",
    "germany":"DE","de":"DE","deutschland":"DE",
    "european union":"EU","eu":"EU",
    "china":"CN","cn":"CN",
    "india":"IN","in":"IN","bharat":"IN",
    "united kingdom":"GB","uk":"GB","england":"GB",
    "lanka":"LK","sri lanka":"LK","sl":"LK",
    "mexico":"MX","mx":"MX",
    "brazil":"BR","br":"BR"
    "canada": "CA", "ca": "CA"
}

# Chapter/heading hints by commodity keywords
_HS_HINTS = {
    # electronics
    "electronics":"85","laptop":"85","phone":"85","modem":"85","router":"85","cable":"85",
    # apparel/textile
    "apparel":"62","clothing":"62","garment":"62","shirt":"62","trousers":"62","jeans":"62","textile":"62","fabric":"62",
    # metals / food
    "brass":"74","honey":"0409",
    # new: fertilizer, tractors
    "fertilizer":"31","fertilisers":"31",
    "tractor":"8701","tractors":"8701"
}

_HS_HINTS.update({
    "egg": "0407",
    "eggs": "0407"
})

_FLAG_WORDS = {
    "controlled":"controlled","dual use":"controlled","dual-use":"controlled",
    "hazmat":"hazmat","dangerous goods":"hazmat","battery":"battery"
}

_VALID_COUNTRIES = ["US","DE","EU","IN","CN","GB","LK","PK","BD","AE","SA","SG","MY","TH","VN","MX","BR"]
_ALIAS_TO_ISO = {**{k:v for k,v in _COUNTRY_MAP.items()}}

_MONTHS = {"jan","feb","mar","apr","may","jun","jul","aug","sep","sept","oct","nov","dec",
           "january","february","march","april","june","july","august","september","october","november","december"}

# -------------------- Helpers --------------------
def _guess_country(token: str):
    return _COUNTRY_MAP.get((token or "").lower().strip())

def _norm_country(s: str):
    if not s: return None
    s = s.strip()
    if s.upper() in _VALID_COUNTRIES or s.upper()=="EU": return s.upper()
    return _ALIAS_TO_ISO.get(s.lower()) or _guess_country(s) or None

def _extract_value_usd(text: str):
    t = (text or "").lower()
    if "$" not in t and " usd" not in t and "usd " not in t: return None
    m = re.search(r'(\$|usd\s*)?([0-9]{1,3}(?:[, ]?[0-9]{3})*(?:\.[0-9]+)?)', t)
    if not m: return None
    try: return float(m.group(2).replace(",","").replace(" ",""))
    except: return None

def _extract_quantity(text: str):
    t = (text or "").lower()
    m = re.search(r'(\d+(?:\.\d+)?)\s*(tons?|t|kg|kgs|kilograms?|pcs?|pieces?)', t)
    if not m: return None, None
    qty = float(m.group(1)); unit = m.group(2)
    unit = {"t":"ton","tons":"ton","kgs":"kg","pcs":"pcs","pieces":"pcs"}.get(unit, unit)
    return qty, unit

def _extract_hs_list(text: str):
    t = text or ""
    hs = []
    for m in re.finditer(r'\b(\d{6,8})\b', t): hs.append(m.group(1))
    for m in re.finditer(r'\b(\d{4}\.\d{2})\b', t): hs.append(m.group(1))
    for m in re.finditer(r'\b(\d{2})\b', t):
        left = t[max(0, m.start()-12):m.start()].lower()
        if any(mon in left.split() for mon in _MONTHS): continue
        hs.append(m.group(1))
    return sorted(set(hs), key=lambda x: (-len(x), x))

def _extract_packaging(text:str):
    t = (text or "").lower()
    if "wood" in t or "crat" in t or "pallet" in t: return "wood"
    return "standard"

def _extract_flags(text:str):
    t = (text or "").lower(); flags = set()
    for k,v in _FLAG_WORDS.items():
        if k in t: flags.add(v)
    return sorted(flags)

def normalize_hs(hs_code: str) -> str:
    """Return normalized HS: 4-digit heading when possible, else 2-digit chapter."""
    if not hs_code: return ""
    clean = re.sub(r"[^\d]", "", hs_code)
    if not clean: return ""
    if len(clean) >= 4: return clean[:4]
    if len(clean) >= 2: return clean[:2]
    return clean

def hs_prefixes(hs_code: str):
    """Return prefixes for rules: [chapter(2), heading(4) if available]."""
    code = normalize_hs(hs_code)
    if not code: return []
    out = []
    if len(code) >= 2: out.append(code[:2])
    if len(code) >= 4: out.append(code[:4])
    return out

def _is_valid_hs(code: str):
    return bool(re.fullmatch(r"\d{2}|\d{4}|\d{6,8}|\d{4}\.\d{2}", code or ""))

# -------------------- Parsers --------------------
def parse_nl_heuristic(text: str):
    t = (text or "").strip()
    if not t: return {}

    origin = destination = None
    m = re.search(r'\bfrom\s+([a-zA-Z ]+?)\s+to\s+([a-zA-Z ]+)\b', t, flags=re.I)
    if m:
        origin = _guess_country(m.group(1)) or origin
        destination = _guess_country(m.group(2)) or destination
    if not destination:
        m2 = re.search(r'\bto\s+([a-zA-Z ]+)\b', t, flags=re.I)
        if m2: destination = _guess_country(m2.group(1)) or destination
    if not origin or not destination:
        found = []
        for name, code in _COUNTRY_MAP.items():
            if re.search(r'\b'+re.escape(name)+r'\b', t, flags=re.I): found.append(code)
        if not origin and found:
            origin = (found[0] if destination != found[0] else (found[1] if len(found)>1 else None))
        if not destination and len(found) >= 1:
            destination = found[-1]

    qty, qty_unit = _extract_quantity(t)
    value = _extract_value_usd(t)
    hs_list = _extract_hs_list(t)

    hs_prefix = None
    if not hs_list:
        for kw, pref in _HS_HINTS.items():
            if re.search(r'\b'+re.escape(kw)+r'\b', t.lower()):
                hs_prefix = pref; break

    packaging = _extract_packaging(t)
    flags = _extract_flags(t)

    detected = [kw for kw in _HS_HINTS.keys() if kw in t.lower()]
    warnings = []
    if len(detected) > 1:
        warnings.append(f"Multiple commodity hints found: {detected}. Using '{detected[0]}' → HS {_HS_HINTS[detected[0]]}.")

    return {
        "origin": origin, "destination": destination,
        "hs_list": hs_list, "hs_fallback_prefix": hs_prefix,
        "value": value, "quantity": qty, "quantity_unit": qty_unit,
        "packaging": packaging, "flags": flags, "warnings": warnings
    }

def parse_nl_llm(text: str):
    if not _OPENAI: return None
    sys = (
        "You are a precise information extractor for international trade shipments. "
        "Return ONLY a compact JSON object with keys: "
        "{origin, destination, hs_list, hs_fallback_prefix, value, quantity, quantity_unit, packaging, flags, commodity_suggestions, rationale}. "
        "origin/destination must be ISO-like codes from: " + ",".join(_VALID_COUNTRIES) + ". "
        "hs_list may include 2/4/6-digit or dotted 4+2. "
        "hs_fallback_prefix can be a 2 or 4 digit hint when only commodity is clear. "
        "value must be a number only if $ or USD is explicit. "
        "packaging = 'wood' if wood/pallet/crate appears, else 'standard'. "
        "flags ⊆ ['controlled','hazmat','battery']."
    )
    try:
        resp = _OPENAI.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role":"system","content":sys},{"role":"user","content":f"Text: {text}"}],
            temperature=0.1
        )
        data = json.loads(resp.choices[0].message.content.strip())
    except Exception:
        return None

    data["origin"] = _norm_country(data.get("origin"))
    data["destination"] = _norm_country(data.get("destination"))
    data["hs_list"] = [h for h in (data.get("hs_list") or []) if _is_valid_hs(h)]
    hfp = data.get("hs_fallback_prefix")
    if isinstance(hfp, str):
        hfp_clean = re.sub(r"[^\d]", "", hfp)
        data["hs_fallback_prefix"] = hfp_clean[:4] if len(hfp_clean) >= 4 else hfp_clean[:2] if len(hfp_clean)>=2 else None
    else:
        data["hs_fallback_prefix"] = None

    pkg = (data.get("packaging") or "").lower()
    data["packaging"] = "wood" if pkg in ["wood","wooden","pallet","crate"] else "standard"

    # value / qty numeric guards
    try: data["value"] = float(data["value"]) if data.get("value") is not None else None
    except: data["value"] = None
    try: data["quantity"] = float(data["quantity"]) if data.get("quantity") is not None else None
    except: data["quantity"] = None

    data["flags"] = [f for f in (data.get("flags") or []) if f in ["controlled","hazmat","battery"]]
    return data

# -------------------- CSV loaders --------------------
def _clean_json_text(s: str) -> str:
    if not isinstance(s, str): return ""
    return s.replace("“", '"').replace("”", '"').replace("’", "'").strip()

def _safe_read_csv(url_or_path: str, local_fallback: str) -> pd.DataFrame:
    try:
        return pd.read_csv(url_or_path)
    except Exception as e_remote:
        try:
            st.warning(f"Couldn’t fetch remote CSV, using local fallback: {local_fallback}")
            return pd.read_csv(local_fallback)
        except Exception as e_local:
            st.error("Failed to load both remote and local CSV. Showing minimal UI only.")
            with st.expander("Loader errors", expanded=False):
                st.exception(e_remote); st.exception(e_local)
            return pd.DataFrame()

@st.cache_data(ttl=300)
def load_rules() -> pd.DataFrame:
    df = _safe_read_csv(RULES_URL, "rules.csv")
    if df.empty: df = pd.DataFrame(columns=["name","priority","active","condition_json","add_forms"])
    df.columns = [c.strip() for c in df.columns]
    for c in ["name","priority","active","condition_json","add_forms"]:
        if c not in df.columns: df[c] = None
    df["condition_json"] = df["condition_json"].map(_clean_json_text)
    df["add_forms"]      = df["add_forms"].map(_clean_json_text)
    def _parse(txt, default):
        try: return json.loads(txt) if isinstance(txt,str) and txt else default
        except Exception: return default
    df["priority"]       = pd.to_numeric(df["priority"], errors="coerce").fillna(100).astype(int)
    df["active"]         = pd.to_numeric(df["active"], errors="coerce").fillna(1).astype(int)
    df["condition_json"] = df["condition_json"].apply(lambda s: _parse(s, {}))
    df["add_forms"]      = df["add_forms"].apply(lambda s: _parse(s, []))
    return df[df["active"] == 1].sort_values(["priority"]).reset_index(drop=True)

@st.cache_data(ttl=300)
def load_fields() -> pd.DataFrame:
    # kept for future (UI now hides this)
    df = _safe_read_csv(FIELDS_URL, "fields.csv")
    if df.empty: df = pd.DataFrame(columns=["form_code","field_key","label","type","required"])
    df.columns = [c.strip() for c in df.columns]
    for c in ["form_code","field_key","label","type","required"]:
        if c not in df.columns: df[c] = None
    df["required"] = pd.to_numeric(df["required"], errors="coerce").fillna(1).astype(int)
    return df

# -------------------- Header / Intro --------------------
st.button("🔄 Reload rules/fields", on_click=lambda: st.cache_data.clear())
with st.expander("Data sources", expanded=False):
    st.write("Rules source:", RULES_URL)
    st.write("Fields source:", FIELDS_URL)

try:
    rules_df = load_rules()
    fields_df = load_fields()
except Exception as e:
    st.error("Problem loading rules/fields; using empty dataframes so the page still renders.")
    with st.expander("Load error", expanded=False):
        st.exception(e)
    rules_df = pd.DataFrame(columns=["name","priority","active","condition_json","add_forms"])
    fields_df = pd.DataFrame(columns=["form_code","field_key","label","type","required"])

st.markdown("## Compliance Compiler")
st.write(
    "Tell me about your shipment and I’ll list the forms you need.\n"
    "I look at where it’s going, what it is (HS code), and the value.\n"
    "Then I show the required forms and why each one is needed."
)

# -------------------- Natural-language input --------------------
with st.expander("🗣️ Describe your shipment in plain English", expanded=True):
    nl = st.text_area(
        "Example: “We’re importing **fertilizer** from **Brazil** to the **United States**, no price yet.”",
        height=120
    )
    if (not origin or not destination or not hs):
    if nl and nl.strip():
        llm = parse_nl_llm(nl) or {}
        heur = parse_nl_heuristic(nl) or {}
        origin = origin or _norm_country(llm.get("origin") or heur.get("origin"))
        destination = destination or _norm_country(llm.get("destination") or heur.get("destination"))
        if not hs:
            pick = (llm.get("hs_list") or heur.get("hs_list") or [])
            pick = pick[0] if pick else (llm.get("hs_fallback_prefix") or heur.get("hs_fallback_prefix"))
            hs = normalize_hs(pick) if pick else hs
            
    if st.button("Understand my description"):
        llm = parse_nl_llm(nl) or {}
        heur = parse_nl_heuristic(nl) or {}
        merged = {
            "origin": _norm_country(llm.get("origin") or heur.get("origin")),
            "destination": _norm_country(llm.get("destination") or heur.get("destination")),
            "hs_list": llm.get("hs_list") or heur.get("hs_list"),
            "hs_fallback_prefix": llm.get("hs_fallback_prefix") or heur.get("hs_fallback_prefix"),
            "value": llm.get("value") if llm.get("value") is not None else heur.get("value"),
            "quantity": llm.get("quantity") or heur.get("quantity"),
            "quantity_unit": llm.get("quantity_unit") or heur.get("quantity_unit"),
            "packaging": llm.get("packaging") or heur.get("packaging"),
            "flags": llm.get("flags") or heur.get("flags"),
            "commodity_suggestions": llm.get("commodity_suggestions") or [],
            "warnings": (llm.get("warnings") or []) + (heur.get("warnings") or []),
            "rationale": llm.get("rationale") or []
        }
        st.session_state.nl_result = merged
        st.success("Parsed! Scroll down to see values filled in below.")
        st.json(merged)

nl_result = st.session_state.get("nl_result", {}) if "nl_result" in st.session_state else {}

if nl_result:
    chips = []
    if nl_result.get("origin"): chips.append(f"Origin: {nl_result['origin']}")
    if nl_result.get("destination"): chips.append(f"Destination: {nl_result['destination']}")
    if nl_result.get("hs_list"): chips.append(f"HS: {normalize_hs(nl_result['hs_list'][0])}")
    elif nl_result.get("hs_fallback_prefix"): chips.append(f"HS hint: {normalize_hs(nl_result['hs_fallback_prefix'])}")
    if chips:
        st.success(" • ".join(chips))

# -------------------- Shipment details --------------------
st.markdown("### Shipment details")
st.info(
    "• **Origin**: where the goods leave from.\n"
    "• **Destination**: where the goods arrive.\n"
    "• **HS Code**: 4-digit heading preferred (e.g., 8701). 2-digit chapter is okay (e.g., 31).\n"
    "• **Invoice Value (USD)**: used for thresholds like EEI."
)

# AI/heuristic-driven defaults (no hard electronics default)
pref_origin      = nl_result.get("origin") or ""
pref_destination = nl_result.get("destination") or ""
# Choose explicit HS if provided, else fallback prefix from commodity
pref_hs_raw = (nl_result.get("hs_list") or [None])[0] or nl_result.get("hs_fallback_prefix")
pref_hs = normalize_hs(pref_hs_raw) if pref_hs_raw else ""

pref_value = nl_result.get("value") if nl_result.get("value") is not None else 0.0

col1, col2, col3 = st.columns(3)
with col1:
    origin_input = st.text_input("Origin (country name or code)", value=pref_origin)
    hs_input = st.text_input("HS Code (4-digit heading or 2-digit chapter)", value=pref_hs)
with col2:
    destination_input = st.text_input("Destination (country name or code)", value=pref_destination)
    value = st.number_input("Invoice Value (USD)", min_value=0.0, value=float(pref_value), step=100.0)
with col3:
    st.write("")
    st.write("")

origin = _norm_country(origin_input)
destination = _norm_country(destination_input)
hs = normalize_hs(hs_input)

st.caption("Please review these details before compiling.")

# If user skipped the “Understand” step, try to parse silently at compile time
def _auto_fill_from_nl_if_needed(nl_text: str):
    global origin, destination, hs
    if origin and destination and hs:
        return
    if not nl_text:
        return
    llm = parse_nl_llm(nl_text) or {}
    heur = parse_nl_heuristic(nl_text) or {}
    origin  = origin  or _norm_country(llm.get("origin") or heur.get("origin"))
    destination = destination or _norm_country(llm.get("destination") or heur.get("destination"))
    if not hs:
        picked = (llm.get("hs_list") or heur.get("hs_list") or [])
        picked = picked[0] if picked else (llm.get("hs_fallback_prefix") or heur.get("hs_fallback_prefix"))
        hs = normalize_hs(picked) if picked else hs

# -------------------- Rule helpers --------------------
def cond_true(cond: dict, payload: dict) -> bool:
    # origin exact
    if "origin" in cond and payload.get("origin") != cond["origin"]:
        return False
    # destination in list
    if "destination_in" in cond and payload.get("destination") not in cond["destination_in"]:
        return False
    # HS prefix match (chapter or heading)
    if "hs_prefix_in" in cond:
        prefs = payload.get("hs_prefixes", [])
        targets = cond["hs_prefix_in"]
        if not any(any(p.startswith(t) or p == t for p in prefs) for t in targets):
            return False
    # value threshold
    if "value_gt" in cond:
        v = payload.get("value")
        if v is None or float(v) <= float(cond["value_gt"]):
            return False
    # packaging equality
    if "packaging_eq" in cond and payload.get("packaging") != cond["packaging_eq"]:
        return False
    # flags containment
    if "flags_has" in cond and cond["flags_has"] not in payload.get("flags", []):
        return False
    return True

# -------------------- Compile --------------------
if st.button("Compile requirements", type="primary"):
    # auto-parse if needed (so fertilizer/brazil/us works without clicking Understand)
    _auto_fill_from_nl_if_needed(nl)

    payload = {
        "origin": origin,
        "destination": destination,
        "hs": hs,
        "value": value,
        "packaging": nl_result.get("packaging") or "standard",
        "flags": nl_result.get("flags") or [],
        "hs_prefixes": hs_prefixes(hs),
    }

    # Base forms
    required = {"CI", "PL"}
    reasons_by_form = {"CI": ["Core document"], "PL": ["Core document"]}

    # Apply rules (ensure your electronics rule uses {"hs_prefix_in":["85"]})
    if not rules_df.empty:
        for _, r in rules_df.iterrows():
            try:
                cond = r["condition_json"]; adds = r["add_forms"]
                if cond_true(cond, payload):
                    for f in adds:
                        required.add(f)
                        reasons_by_form.setdefault(f, []).append(f'Rule: {r.get("name","(unnamed)")}')
            except Exception:
                st.warning(f"Skipped a bad rule row: {getattr(r,'name','(no name)')}")
    
    # Hard stop if still missing essentials (prevents irrelevant outputs)
    missing = []
    if not origin: missing.append("origin")
    if not destination: missing.append("destination")
    if not hs: missing.append("HS code (chapter or 4-digit heading)")
    
    if missing:
        st.error("I couldn’t extract: " + ", ".join(missing) + ". "
                 "Please click **Understand my description** or fill the fields manually.")
        st.stop()

    else:
        st.info("No active rules loaded — showing only core forms (CI, PL).")

    # Results
    st.subheader("✅ Required forms and their rationale")
    for i, f in enumerate(sorted(required), start=1):
        reason = "; ".join(reasons_by_form.get(f, ["Matched rule(s)"]))
        st.write(f"{i}. **{f}**: {reason}")

    # Generate documents (CI/PL only)
    if PDF_OK:
        st.subheader("📄 Generate documents")
        items_ci = [{
            "description": "Line item",
            "hs_code": hs or "",
            "qty": 1, "unit": "pcs",
            "unit_price": f"{value:,.2f}", "amount": f"{value:,.2f}",
        }]
        subtot = value; freight = 0.0; insurance = 0.0; total = subtot + freight + insurance
        base_data = {
            "exporter_name": "Exporter (enter on final form)",
            "consignee_name": "Consignee (enter on final form)",
            "origin": origin or "",
            "destination": destination or "",
            "incoterm": "DAP",
            "currency": "USD",
            "invoice_no": "INV-001",
            "invoice_date": "",
            "subtotal": f"{subtot:,.2f}",
            "freight": f"{freight:,.2f}",
            "insurance": f"{insurance:,.2f}",
            "total": f"{total:,.2f}",
            "packages_count": 1,
            "gross_weight_kg": "",
            "net_weight_kg": "",
            "shipment_ref": "INV-001",
        }
        if "CI" in required:
            ci_buf = make_ci_pdf(base_data, items_ci)
            st.download_button(
                "⬇️ Download Commercial Invoice (PDF)",
                data=ci_buf.getvalue(),
                file_name=f"{base_data['invoice_no']}_Commercial_Invoice.pdf",
                mime="application/pdf"
            )
        if "PL" in required:
            pl_rows = [{"pkg_no": 1, "description": "Line item", "qty": 1, "unit": "pcs", "gross_wt": "", "net_wt": ""}]
            pl_buf = make_pl_pdf(base_data, pl_rows)
            st.download_button(
                "⬇️ Download Packing List (PDF)",
                data=pl_buf.getvalue(),
                file_name=f"{base_data['invoice_no']}_Packing_List.pdf",
                mime="application/pdf"
            )
    else:
        st.info("Document generation not available (pdf_utils.py not found).")
