# app.py — Streamlit app with simplified UI and clearer wording

import json, re
import pandas as pd
import streamlit as st

st.set_page_config(page_title="Compliance Compiler", page_icon="🌍")

# ---- CONFIG
RULES_URL = st.secrets.get("RULES_CSV_URL", "rules.csv")
FIELDS_URL = st.secrets.get("FIELDS_CSV_URL", "fields.csv")

# ---- Optional PDF support
try:
    from pdf_utils import make_ci_pdf, make_pl_pdf  # removed make_simple_statement use
    PDF_OK = True
except Exception:
    PDF_OK = False

# ---------- Natural-language extraction (your existing logic)
try:
    from openai import OpenAI
    _OPENAI = OpenAI(api_key=st.secrets.get("OPENAI_API_KEY")) if st.secrets.get("OPENAI_API_KEY") else None
except Exception:
    _OPENAI = None

_COUNTRY_MAP = {
    "united states":"US","usa":"US","us":"US","america":"US",
    "germany":"DE","de":"DE","deutschland":"DE",
    "european union":"EU","eu":"EU",
    "china":"CN","cn":"CN","india":"IN","in":"IN",
    "united kingdom":"GB","uk":"GB","england":"GB",
    "lanka":"LK","sri lanka":"LK","sl":"LK"
}
_HS_HINTS = {
    "electronics":"85","laptop":"85","phone":"85","modem":"85","router":"85","cable":"85",
    "apparel":"62","clothing":"62","garment":"62","shirt":"62","trousers":"62","jeans":"62",
    "textile":"62","fabric":"62",
    "brass":"74","honey":"0409"
}
_FLAG_WORDS = {"controlled":"controlled","dual use":"controlled","dual-use":"controlled",
               "hazmat":"hazmat","dangerous goods":"hazmat","battery":"battery"}

def _guess_country(token:str):
    return _COUNTRY_MAP.get(token.lower().strip())

def _extract_value_usd(text: str):
    t = text.lower()
    if "$" not in t and " usd" not in t and "usd " not in t: return None
    m = re.search(r'(\$|usd\s*)?([0-9]{1,3}(?:[, ]?[0-9]{3})*(?:\.[0-9]+)?)', t)
    if not m: return None
    try: return float(m.group(2).replace(",","").replace(" ",""))
    except: return None

def _extract_quantity(text: str):
    t = text.lower()
    m = re.search(r'(\d+(?:\.\d+)?)\s*(tons?|t|kg|kgs|kilograms?|pcs?|pieces?)', t)
    if not m: return None, None
    qty = float(m.group(1)); unit = m.group(2)
    unit = {"t":"ton","tons":"ton","kgs":"kg","pcs":"pcs","pieces":"pcs"}.get(unit, unit)
    return qty, unit

_MONTHS = {"jan","feb","mar","apr","may","jun","jul","aug","sep","sept","oct","nov","dec",
           "january","february","march","april","june","july","august","september","october","november","december"}

def _extract_hs_list(text: str):
    t = text
    hs = []
    for m in re.finditer(r'\b(\d{6,8})\b', t): hs.append(m.group(1))
    for m in re.finditer(r'\b(\d{4}\.\d{2})\b', t): hs.append(m.group(1))
    for m in re.finditer(r'\b(\d{2})\b', t):
        left = t[max(0, m.start()-12):m.start()].lower()
        if any(mon in left.split() for mon in _MONTHS): continue
        hs.append(m.group(1))
    return sorted(set(hs), key=lambda x: (-len(x), x))

def _extract_packaging(text:str):
    t = text.lower()
    if "wood" in t or "crat" in t or "pallet" in t: return "wood"
    return "standard"

def _extract_flags(text:str):
    t = text.lower(); flags = set()
    for k,v in _FLAG_WORDS.items():
        if k in t: flags.add(v)
    return sorted(flags)

def parse_nl_heuristic(text: str):
    t = text.strip()
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
    return {"origin":origin,"destination":destination,"hs_list":hs_list,
            "hs_fallback_prefix":hs_prefix,"value":value,"quantity":qty,"quantity_unit":qty_unit,
            "packaging":packaging,"flags":flags,"warnings":warnings}

_VALID_COUNTRIES = ["US","DE","EU","IN","CN","GB","LK","PK","BD","AE","SA","SG","MY","TH","VN"]
_ALIAS_TO_ISO = {"lanka":"LK","sri lanka":"LK","sl":"LK","india":"IN","bharat":"IN","germany":"DE","deutschland":"DE",
                 "united states":"US","usa":"US","america":"US","u.s.":"US","uk":"GB","england":"GB","china":"CN","prc":"CN"}

def _norm_country(s: str):
    if not s: return None
    s = s.strip().lower()
    if s in _ALIAS_TO_ISO: return _ALIAS_TO_ISO[s]
    if s.upper() in _VALID_COUNTRIES: return s.upper()
    return None

def _is_valid_hs(code: str):
    return bool(re.fullmatch(r"\d{2}|\d{4}|\d{6,8}|\d{4}\.\d{2}", code))

def parse_nl_llm(text: str):
    if not _OPENAI: return None
    sys = ("You are a precise information extractor for international trade shipments. "
           "Return ONLY a compact JSON object with keys: "
           "{origin, destination, hs_list, hs_fallback_prefix, value, currency, quantity, quantity_unit, packaging, flags, commodity_suggestions, rationale}. "
           "Use allowlist: " + ",".join(_VALID_COUNTRIES) + ".")
    try:
        resp = _OPENAI.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role":"system","content":sys},{"role":"user","content":f'Text: {text}'}],
            temperature=0.1
        )
        data = json.loads(resp.choices[0].message.content.strip())
    except Exception:
        return None
    data["origin"] = _norm_country(data.get("origin"))
    data["destination"] = _norm_country(data.get("destination"))
    data["hs_list"] = [h for h in (data.get("hs_list") or []) if isinstance(h,str) and _is_valid_hs(h)]
    hfp = data.get("hs_fallback_prefix")
    data["hs_fallback_prefix"] = hfp if (isinstance(hfp,str) and _is_valid_hs(hfp)) else hfp if isinstance(hfp,str) and len(hfp) in (2,4) else None
    data["packaging"] = "wood" if (data.get("packaging","").lower() in ["wood","wooden","pallet","crate"]) else "standard"
    try: data["value"] = float(data["value"]) if data.get("value") is not None else None
    except: data["value"] = None
    try: data["quantity"] = float(data["quantity"]) if data.get("quantity") is not None else None
    except: data["quantity"] = None
    data["flags"] = [f for f in (data.get("flags") or []) if f in ["controlled","hazmat","battery"]]
    return data

# ---- Safe CSV loader
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
    # kept for future, but we won’t render it now
    df = _safe_read_csv(FIELDS_URL, "fields.csv")
    if df.empty: df = pd.DataFrame(columns=["form_code","field_key","label","type","required"])
    df.columns = [c.strip() for c in df.columns]
    for c in ["form_code","field_key","label","type","required"]:
        if c not in df.columns: df[c] = None
    df["required"] = pd.to_numeric(df["required"], errors="coerce").fillna(1).astype(int)
    return df

# ---- Always-visible controls
st.button("🔄 Reload rules/fields", on_click=lambda: st.cache_data.clear())
with st.expander("Data sources", expanded=False):
    st.write("Rules source:", RULES_URL)
    st.write("Fields source:", FIELDS_URL)

# ---- Load data
try:
    rules_df = load_rules()
    fields_df = load_fields()
except Exception as e:
    st.error("Problem loading rules/fields; using empty dataframes so the page still renders.")
    with st.expander("Load error", expanded=False):
        st.exception(e)
    rules_df = pd.DataFrame(columns=["name","priority","active","condition_json","add_forms"])
    fields_df = pd.DataFrame(columns=["form_code","field_key","label","type","required"])

# ---- Header + friendly intro
st.markdown("## Compliance Compiler")
st.write(
    "Tell me about your shipment and I’ll list the forms you need.\n"
    "I look at where it’s going, what it is (HS code), and the value.\n"
    "Then I show the required forms and why each one is needed."
)

# ==== Natural-language input (plain English -> structured) ====
with st.expander("🗣️ Describe your shipment in plain English", expanded=True):
    nl = st.text_area(
        "Example: “We’re shipping **IP modems** from the **US** to **Germany**, $12,000, on pallets.”",
        height=120
    )
    if st.button("Understand my description"):
        llm = parse_nl_llm(nl) or {}
        heur = parse_nl_heuristic(nl) or {}
        merged = {
            "origin": llm.get("origin") or heur.get("origin"),
            "destination": llm.get("destination") or heur.get("destination"),
            "hs_list": llm.get("hs_list") or heur.get("hs_list"),
            "hs_fallback_prefix": llm.get("hs_fallback_prefix") or heur.get("hs_fallback_prefix"),
            "value": llm.get("value", None) if llm.get("value", None) is not None else heur.get("value"),
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

# ---- Input explanation (what each input means)
st.markdown("### Shipment details")
st.info(
    "• **Origin**: where the goods leave from.\n"
    "• **Destination**: where the goods arrive.\n"
    "• **HS Code**: the product classification. Chapter/heading (e.g., 85 or 8517.62) is okay.\n"
    "• **Invoice Value (USD)**: the shipment price. Used for thresholds like EEI."
)

# Defaults from NL
countries = ["US","DE","EU","IN","CN","GB","LK"]
pref_origin      = nl_result.get("origin") or "US"
pref_destination = nl_result.get("destination") or "DE"
pref_hs = (nl_result.get("hs_list") or [None])[0] or nl_result.get("hs_fallback_prefix") or "8517.62"
pref_value = nl_result.get("value") if nl_result.get("value") is not None else 12000.0

# ——— Minimal input UI (packaging/flags now fixed and hidden)
col1, col2, col3 = st.columns(3)
with col1:
    origin = st.selectbox("Origin", countries, index=countries.index(pref_origin) if pref_origin in countries else 0)
    hs = st.text_input("HS Code or prefix", pref_hs)
with col2:
    destination = st.selectbox("Destination", countries, index=countries.index(pref_destination) if pref_destination in countries else 1)
    value = st.number_input("Invoice Value (USD)", min_value=0.0, value=float(pref_value), step=100.0)
with col3:
    st.write("")  # spacer
    st.write("")

# Fixed (hidden) settings to keep UX simple
packaging = nl_result.get("packaging") or "standard"
flags = nl_result.get("flags") or []

st.caption("Please review these details before compiling.")

# ---- Rule helpers
def hs_prefixes(hs_code: str):
    hs_code = (hs_code or "").replace(" ", "")
    parts = re.split(r"[.\-]", hs_code)
    out = []
    if len(parts) >= 1 and parts[0]: out.append(parts[0])
    if len(parts) >= 2:
        out.append(parts[0] + parts[1])
        out.append(parts[0] + "." + parts[1])
    if len(parts) >= 3:
        out.append(parts[0] + parts[1] + parts[2])
        out.append(parts[0] + "." + parts[1] + "." + parts[2])
        out.append(parts[0] + parts[1] + "." + parts[2])
    seen, res = set(), []
    for p in out:
        if p not in seen: res.append(p); seen.add(p)
    return res

def cond_true(cond: dict, payload: dict) -> bool:
    if "origin" in cond and payload.get("origin") != cond["origin"]: return False
    if "destination_in" in cond and payload.get("destination") not in cond["destination_in"]: return False
    if "hs_prefix_in" in cond:
        prefs = payload.get("hs_prefixes", []); targets = cond["hs_prefix_in"]
        if not any(any(p.startswith(t) or p == t for p in prefs) for t in targets): return False
    if "value_gt" in cond:
        v = payload.get("value"); if v is None or float(v) <= float(cond["value_gt"]): return False
    if "packaging_eq" in cond and packaging != cond["packaging_eq"]: return False
    if "flags_has" in cond and cond["flags_has"] not in flags: return False
    return True

# ---- Compile
if st.button("Compile requirements", type="primary"):
    payload = {
        "origin": origin,
        "destination": destination,
        "hs": hs,
        "value": value,
        "packaging": packaging,     # fixed behind the scenes
        "flags": flags,             # fixed behind the scenes
        "hs_prefixes": hs_prefixes(hs),
    }

    # Base set
    required = {"CI", "PL"}
    # Track reasons per form (form -> [reasons])
    reasons_by_form = {"CI": ["Core document"], "PL": ["Core document"]}

    # Apply rules
    if not rules_df.empty:
        for _, r in rules_df.iterrows():
            try:
                cond = r["condition_json"]; adds = r["add_forms"]
                if cond_true(cond, payload):
                    for f in adds:
                        required.add(f)
                        reasons_by_form.setdefault(f, []).append(f'Rule: {r["name"]}')
            except Exception:
                st.warning(f"Skipped a bad rule row: {getattr(r,'name','(no name)')}")
    else:
        st.info("No active rules loaded — showing only core forms (CI, PL).")

    # ---- Results: Required forms and their rationale
    st.subheader("✅ Required forms and their rationale")
    for i, f in enumerate(sorted(required), start=1):
        reason = "; ".join(reasons_by_form.get(f, ["Matched rule(s)"]))
        st.write(f"{i}. **{f}**: {reason}")

    # (Removed “Fields to capture” per request)

    # ---- Generate documents (CI / PL only)
    if PDF_OK:
        st.subheader("📄 Generate documents")
        # Minimal dummy CI line item
        items_ci = [{
            "description": "IP Modem",
            "hs_code": hs,
            "qty": 1,
            "unit": "pcs",
            "unit_price": f"{value:,.2f}",
            "amount": f"{value:,.2f}",
        }]
        subtot = value; freight = 0.0; insurance = 0.0; total = subtot + freight + insurance
        base_data = {
            "exporter_name": "Exporter (enter on final form)",
            "consignee_name": "Consignee (enter on final form)",
            "origin": origin,
            "destination": destination,
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
                "⬇️ Download Commercial Invoice (Document)",
                data=ci_buf.getvalue(),
                file_name=f"{base_data['invoice_no']}_Commercial_Invoice.pdf",
                mime="application/pdf"
            )
        if "PL" in required:
            pl_rows = [{"pkg_no": 1, "description": "IP Modem", "qty": 1, "unit": "pcs", "gross_wt": "", "net_wt": ""}]
            pl_buf = make_pl_pdf(base_data, pl_rows)
            st.download_button(
                "⬇️ Download Packing List (Document)",
                data=pl_buf.getvalue(),
                file_name=f"{base_data['invoice_no']}_Packing_List.pdf",
                mime="application/pdf"
            )
    else:
        st.info("Document generation not available (pdf_utils.py not found).")
