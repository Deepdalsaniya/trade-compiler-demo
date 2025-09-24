# app.py — Safe, end-to-end Streamlit app (Sheets + fallback + UI + compile + optional PDFs)

import json, re
import pandas as pd
import streamlit as st

st.set_page_config(page_title="Compliance Compiler (MVP)", page_icon="🌍")

# ---- CONFIG / CONSTANTS (must come before loaders & UI that reference them)
RULES_URL = st.secrets.get("RULES_CSV_URL", "rules.csv")
FIELDS_URL = st.secrets.get("FIELDS_CSV_URL", "fields.csv")

# ---- Optional PDF support (safe if pdf_utils.py is missing)
try:
    from pdf_utils import make_ci_pdf, make_pl_pdf, make_simple_statement
    PDF_OK = True
except Exception:
    PDF_OK = False

# ---------- Natural-language extraction (two layers)
import re, json

# Optional LLM client
try:
    from openai import OpenAI
    _OPENAI = OpenAI(api_key=st.secrets.get("OPENAI_API_KEY")) if st.secrets.get("OPENAI_API_KEY") else None
except Exception:
    _OPENAI = None

_COUNTRY_MAP = {
    "united states":"US","usa":"US","us":"US","america":"US",
    "germany":"DE","de":"DE","deutschland":"DE",
    "european union":"EU","eu":"EU",
    "china":"CN","cn":"CN","india":"IN","in":"IN","united kingdom":"GB","uk":"GB"
}
_HS_HINTS = {
    # broad fallbacks when no HS in text
    "electronics":"85","laptop":"85","phone":"85","modem":"85","router":"85","cable":"85",
    "apparel":"62","clothing":"62","garment":"62","shirt":"62","trousers":"62","jeans":"62",
    "textile":"62","fabric":"62"
}
_FLAG_WORDS = {
    "controlled":"controlled","dual use":"controlled","dual-use":"controlled",
    "hazmat":"hazmat","dangerous goods":"hazmat","battery":"battery"
}

def _guess_country(token:str):
    t = token.lower().strip()
    return _COUNTRY_MAP.get(t)

def _extract_value_usd(text: str):
    """
    Only return a USD value if currency markers are present: '$' or 'usd'.
    Avoid capturing plain numbers like '2 ton' or dates like 'Dec 10'.
    """
    t = text.lower()
    # Require currency marker somewhere nearby
    if "$" not in t and " usd" not in t and "usd " not in t:
        return None
    m = re.search(r'(\$|usd\s*)?([0-9]{1,3}(?:[, ]?[0-9]{3})*(?:\.[0-9]+)?)', t)
    if not m: 
        return None
    num = m.group(2).replace(",", "").replace(" ", "")
    try:
        return float(num)
    except:
        return None

def _extract_quantity(text: str):
    """
    Extract a shipment quantity + unit if present (e.g., '2 ton', '15 pcs').
    Not used by rules yet, but helpful for PDFs and UX.
    """
    t = text.lower()
    m = re.search(r'(\d+(?:\.\d+)?)\s*(tons?|t|kg|kgs|kilograms?|pcs?|pieces?)', t)
    if not m:
        return None, None
    qty = float(m.group(1))
    unit = m.group(2)
    # normalize units
    unit = {"t":"ton","tons":"ton","kgs":"kg","pcs":"pcs","pieces":"pcs"}.get(unit, unit)
    return qty, unit

_MONTHS = {"jan","feb","mar","apr","may","jun","jul","aug","sep","sept","oct","nov","dec",
           "january","february","march","april","june","july","august","september","october","november","december"}

def _extract_hs_list(text: str):
    """
    Find HS-like tokens but avoid picking up dates (e.g., 'Dec 10').
    Accept:
      - '8517.62'
      - '851762'
      - 2-digit only if NOT right after a month word.
    """
    t = text
    hs = []

    # 6+ digit contiguous
    for m in re.finditer(r'\b(\d{6,8})\b', t):
        hs.append(m.group(1))
    # 4+2 dotted
    for m in re.finditer(r'\b(\d{4}\.\d{2})\b', t):
        hs.append(m.group(1))
    # careful 2-digit
    for m in re.finditer(r'\b(\d{2})\b', t):
        two = m.group(1)
        # skip if looks like a day after a month word within 2 tokens left
        left = t[max(0, m.start()-12):m.start()].lower()
        if any(mon in left.split() for mon in _MONTHS):
            continue
        hs.append(two)

    # prefer longer codes, then unique
    hs = sorted(set(hs), key=lambda x: (-len(x), x))
    return hs


def _extract_packaging(text:str):
    t = text.lower()
    if "wood" in t or "crat" in t: return "wood"
    if "pallet" in t: return "wood"  # good enough for demo
    return "standard"

def _extract_flags(text:str):
    t = text.lower()
    flags = set()
    for k,v in _FLAG_WORDS.items():
        if k in t: flags.add(v)
    return sorted(flags)

# extend hints
_HS_HINTS.update({
    "brass":"74",          # Copper & articles (brass alloys fall under Chapter 74)
    "honey":"0409"         # Natural honey
})

def parse_nl_heuristic(text: str):
    t = text.strip()
    if not t:
        return {}

    # origin / destination
    origin = destination = None

    # explicit "from X to Y"
    m = re.search(r'\bfrom\s+([a-zA-Z ]+?)\s+to\s+([a-zA-Z ]+)\b', t, flags=re.I)
    if m:
        origin = _guess_country(m.group(1)) or origin
        destination = _guess_country(m.group(2)) or destination

    # explicit "to X" (destination only)
    if not destination:
        m2 = re.search(r'\bto\s+([a-zA-Z ]+)\b', t, flags=re.I)
        if m2:
            destination = _guess_country(m2.group(1)) or destination

    # scan all tokens for countries as fallback
    if not origin or not destination:
        found = []
        for name, code in _COUNTRY_MAP.items():
            if re.search(r'\b'+re.escape(name)+r'\b', t, flags=re.I):
                found.append(code)
        if not origin and found:
            # if we already assigned destination from "to", use next for origin
            origin = (found[0] if destination != found[0] else (found[1] if len(found) > 1 else None))
        if not destination and len(found) >= 1:
            destination = found[-1]

    # quantities & value
    qty, qty_unit = _extract_quantity(t)
    value = _extract_value_usd(t)  # will be None for "2 ton" which is correct

    # HS codes
    hs_list = _extract_hs_list(t)

    # commodity-inferred HS if nothing explicit
    hs_prefix = None
    if not hs_list:
        for kw, pref in _HS_HINTS.items():
            if re.search(r'\b'+re.escape(kw)+r'\b', t.lower()):
                hs_prefix = pref
                break

    packaging = _extract_packaging(t)
    flags = _extract_flags(t)

    # ambiguity note if multiple commodity keywords present
    detected = [kw for kw in _HS_HINTS.keys() if kw in t.lower()]
    warnings = []
    if len(detected) > 1:
        warnings.append(f"Multiple commodity hints found: {detected}. Using '{detected[0]}' → HS { _HS_HINTS[detected[0]] }.")

    return {
        "origin": origin,
        "destination": destination,
        "hs_list": hs_list,                 # may be []
        "hs_fallback_prefix": hs_prefix,    # e.g., '74' for brass or '0409' for honey
        "value": value,                     # USD only when currency markers present
        "quantity": qty,
        "quantity_unit": qty_unit,
        "packaging": packaging,
        "flags": flags,
        "warnings": warnings
    }


# ---------- LLM-first extraction with normalization & validation
_VALID_COUNTRIES = ["US","DE","EU","IN","CN","GB","LK","PK","BD","AE","SA","SG","MY","TH","VN"]
_ALIAS_TO_ISO = {
    "lanka":"LK","sri lanka":"LK","sl":"LK",
    "india":"IN","bharat":"IN",
    "germany":"DE","deutschland":"DE","frg":"DE",
    "united states":"US","usa":"US","america":"US","u.s.":"US",
    "uk":"GB","united kingdom":"GB","england":"GB",
    "china":"CN","prc":"CN","mainland china":"CN"
}
# keep your existing _HS_HINTS (now LLM can add suggestions too)

def _norm_country(s: str):
    if not s: return None
    s = s.strip().lower()
    if s in _ALIAS_TO_ISO: return _ALIAS_TO_ISO[s]
    if s.upper() in _VALID_COUNTRIES: return s.upper()
    return None  # unknown

def _is_valid_hs(code: str):
    # Accept 2, 4, 6, or dotted 4+2
    if re.fullmatch(r"\d{2}", code): return True
    if re.fullmatch(r"\d{4}", code): return True
    if re.fullmatch(r"\d{6,8}", code): return True
    if re.fullmatch(r"\d{4}\.\d{2}", code): return True
    return False

def parse_nl_llm(text: str):
    if not _OPENAI:
        return None
    sys = (
        "You are a precise information extractor for international trade shipments. "
        "Return ONLY a compact JSON object with keys: "
        "{origin, destination, hs_list, hs_fallback_prefix, value, currency, quantity, quantity_unit, packaging, flags, commodity_suggestions, rationale}. "
        "Rules: "
        "1) origin/destination as country codes from this allowlist: " + ",".join(_VALID_COUNTRIES) + ". "
        "   If text says 'to Lanka', set destination='LK'. If you cannot determine, use null. "
        "2) hs_list: array of strings like '8517.62' or '0409' or '74'. Only include if clearly stated or strongly implied; otherwise empty. "
        "3) hs_fallback_prefix: a 2–4 digit hint like '85' or '0409' when commodity is clear but HS not given. "
        "4) value: number ONLY if currency is explicit ($ or USD). Otherwise null. "
        "5) quantity/quantity_unit: capture phrases like '2 ton', '15 pcs' when present. "
        "6) packaging: 'wood' if wood/wooden/pallet/crate is mentioned, else 'standard'. "
        "7) flags: subset of ['controlled','hazmat','battery'] if clearly suggested. "
        "8) commodity_suggestions: short array of strings like ['brass','honey'] if multiple commodities mentioned. "
        "9) rationale: short bullet strings explaining choices."
    )
    user = f"Text: {text}"
    try:
        resp = _OPENAI.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role":"system","content":sys},{"role":"user","content":user}],
            temperature=0.1
        )
        content = resp.choices[0].message.content.strip()
        data = json.loads(content)
    except Exception:
        return None

    # Post-process & validate
    data["origin"] = _norm_country(data.get("origin"))
    data["destination"] = _norm_country(data.get("destination"))

    # Clean HS list
    hs_list = []
    for h in (data.get("hs_list") or []):
        if isinstance(h, str) and _is_valid_hs(h):
            hs_list.append(h)
    data["hs_list"] = hs_list

    # Fallback: if no hs_list, allow 2–4 digit prefix suggestion (e.g., '74' brass, '0409' honey)
    hfp = data.get("hs_fallback_prefix")
    if hfp and not _is_valid_hs(hfp):
        hfp = None
    data["hs_fallback_prefix"] = hfp

    # Normalize packaging
    pkg = (data.get("packaging") or "").lower()
    data["packaging"] = "wood" if pkg in ["wood","wooden","pallet","crate"] else "standard"

    # Guard value: must be number
    try:
        v = data.get("value", None)
        data["value"] = float(v) if v is not None else None
    except:
        data["value"] = None

    # Quantity
    try:
        q = data.get("quantity", None)
        data["quantity"] = float(q) if q is not None else None
    except:
        data["quantity"] = None

    # Flags keep as array
    data["flags"] = [f for f in (data.get("flags") or []) if f in ["controlled","hazmat","battery"]]

    return data

def _clean_json_text(s: str) -> str:
    if not isinstance(s, str):
        return ""
    # replace smart quotes and trim
    return s.replace("“", '"').replace("”", '"').replace("’", "'").strip()

def _safe_read_csv(url_or_path: str, local_fallback: str) -> pd.DataFrame:
    """Try remote; if it fails, use local; if that fails, return empty DataFrame."""
    try:
        return pd.read_csv(url_or_path)
    except Exception as e_remote:
        try:
            st.warning(f"Couldn’t fetch remote CSV, using local fallback: {local_fallback}")
            return pd.read_csv(local_fallback)
        except Exception as e_local:
            st.error("Failed to load both remote and local CSV. Showing minimal UI only.")
            with st.expander("Loader errors", expanded=False):
                st.exception(e_remote)
                st.exception(e_local)
            return pd.DataFrame()

# ---- 3) CACHED LOADERS
@st.cache_data(ttl=300)
def load_rules() -> pd.DataFrame:
    df = _safe_read_csv(RULES_URL, "rules.csv")
    if df.empty:
        df = pd.DataFrame(columns=["name","priority","active","condition_json","add_forms"])

    # normalize columns and ensure presence
    df.columns = [c.strip() for c in df.columns]
    for c in ["name","priority","active","condition_json","add_forms"]:
        if c not in df.columns:
            df[c] = None

    # clean & parse json-ish columns
    df["condition_json"] = df["condition_json"].map(_clean_json_text)
    df["add_forms"]      = df["add_forms"].map(_clean_json_text)

    def _parse(txt, default):
        try:
            return json.loads(txt) if isinstance(txt, str) and txt else default
        except Exception:
            return default

    df["priority"]       = pd.to_numeric(df["priority"], errors="coerce").fillna(100).astype(int)
    df["active"]         = pd.to_numeric(df["active"], errors="coerce").fillna(1).astype(int)
    df["condition_json"] = df["condition_json"].apply(lambda s: _parse(s, {}))
    df["add_forms"]      = df["add_forms"].apply(lambda s: _parse(s, []))

    # keep only active; still safe if empty
    return df[df["active"] == 1].sort_values(["priority"]).reset_index(drop=True)

@st.cache_data(ttl=300)
def load_fields() -> pd.DataFrame:
    df = _safe_read_csv(FIELDS_URL, "fields.csv")
    if df.empty:
        df = pd.DataFrame(columns=["form_code","field_key","label","type","required"])
    df.columns = [c.strip() for c in df.columns]
    for c in ["form_code","field_key","label","type","required"]:
        if c not in df.columns:
            df[c] = None
    df["required"] = pd.to_numeric(df["required"], errors="coerce").fillna(1).astype(int)
    return df

# ---- 4) UI CONTROLS THAT ALWAYS RENDER
st.button("🔄 Reload rules/fields", on_click=lambda: st.cache_data.clear())
with st.expander("Data sources", expanded=False):
    st.write("Rules source:", RULES_URL)
    st.write("Fields source:", FIELDS_URL)

# ---- 5) Load dataframes (after functions exist)
try:
    rules_df = load_rules()
    fields_df = load_fields()
except Exception as e:
    st.error("Problem loading rules/fields; using empty dataframes so the page still renders.")
    with st.expander("Load error", expanded=False):
        st.exception(e)
    rules_df = pd.DataFrame(columns=["name","priority","active","condition_json","add_forms"])
    fields_df = pd.DataFrame(columns=["form_code","field_key","label","type","required"])

# ---- 6) Page title + diagnostics
st.title("🌍 Compliance Compiler (US → EU Demo)")
with st.expander("Diagnostics", expanded=False):
    st.write("Rules loaded:", len(rules_df) if isinstance(rules_df, pd.DataFrame) else 0)
    st.write("Fields loaded:", len(fields_df) if isinstance(fields_df, pd.DataFrame) else 0)
    if not fields_df.empty:
        st.write("Unique form_code values in Fields:", sorted(fields_df["form_code"].dropna().unique().tolist()))
    st.write("PDF generation available:", PDF_OK)
# ==== Natural-language input (plain English -> structured) ====

with st.expander("🗣️ Describe your shipment in plain English", expanded=True):
    nl = st.text_area("Example: “We’re shipping telecom modems from India to Lanka, 2 tons, no price yet.”", height=120)
    if st.button("Understand my description"):
        llm = parse_nl_llm(nl) or {}
        heur = parse_nl_heuristic(nl) or {}

        # prefer LLM answers, fall back to heuristics
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


# Defaults from NL parse (if available)
nl_result = st.session_state.get("nl_result", {}) if "nl_result" in st.session_state else {}

pref_origin      = nl_result.get("origin") or "US"
pref_destination = nl_result.get("destination") or "DE"

pref_hs = None
if nl_result.get("hs_list"):
    pref_hs = nl_result["hs_list"][0]
elif nl_result.get("hs_fallback_prefix"):
    pref_hs = nl_result["hs_fallback_prefix"]
pref_hs = pref_hs or "8517.62"

pref_value = nl_result.get("value")  # can be None
pref_pack  = nl_result.get("packaging") or "standard"
pref_flags = nl_result.get("flags") or []

# Optional display of multiple commodities
for w in nl_result.get("warnings", []):
    st.warning(w)
if nl_result.get("commodity_suggestions"):
    st.info(f"Commodity suggestions: {', '.join(nl_result['commodity_suggestions'])}")


# ---- 7) Core rule helpers
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
    # unique preserve order
    seen, res = set(), []
    for p in out:
        if p not in seen:
            res.append(p); seen.add(p)
    return res

def cond_true(cond: dict, payload: dict) -> bool:
    if "origin" in cond and payload.get("origin") != cond["origin"]: return False
    if "destination_in" in cond and payload.get("destination") not in cond["destination_in"]: return False
    if "hs_prefix_in" in cond:
        prefs = payload.get("hs_prefixes", [])
        targets = cond["hs_prefix_in"]
        ok = any(any(p.startswith(t) or p == t for p in prefs) for t in targets)
        if not ok: return False
    if "value_gt" in cond:
        v = payload.get("value")
        if v is None or float(v) <= float(cond["value_gt"]): return False
    if "packaging_eq" in cond and payload.get("packaging") != cond["packaging_eq"]: return False
    if "flags_has" in cond and cond["flags_has"] not in payload.get("flags", []): return False
    return True

# ---- 8) Minimal input UI (pre-filled from NL when available)
countries = ["US","DE","EU","IN","CN","GB","LK"]

col1, col2, col3 = st.columns(3)
with col1:
    origin = st.selectbox("Origin", countries, index=countries.index(pref_origin) if pref_origin in countries else 0)
    hs = st.text_input("HS Code or prefix", pref_hs)
with col2:
    destination = st.selectbox("Destination", countries, index=countries.index(pref_destination) if pref_destination in countries else 1)
    value = st.number_input("Invoice Value (USD)", min_value=0.0, value=float(pref_value) if pref_value is not None else 0.0, step=100.0)
with col3:
    packaging = st.selectbox("Packaging", ["standard","wood"], index=["standard","wood"].index(pref_pack) if pref_pack in ["standard","wood"] else 0)

flags = st.multiselect("Flags", ["controlled","hazmat","battery"], default=pref_flags)

# Optional: extra fields for PDFs (won't break if PDFs disabled)
with st.expander("Parties & shipment (for PDFs / optional)", expanded=False):
    colA, colB, colC = st.columns(3)
    with colA:
        exporter_name = st.text_input("Exporter Name", "Acme Exporters LLC")
        invoice_no = st.text_input("Invoice No", "INV-001")
    with colB:
        consignee_name = st.text_input("Consignee Name", "Beta GmbH")
        incoterm = st.text_input("Incoterm", "DAP")
    with colC:
        currency = st.text_input("Currency", "USD")

# ---- 9) Compile button
if st.button("Compile requirements", type="primary"):
    payload = {
        "origin": origin,
        "destination": destination,
        "hs": hs,
        "value": value,
        "packaging": packaging,
        "flags": flags,
        "hs_prefixes": hs_prefixes(hs),
    }

    # Start with core
    required = {"CI", "PL"}
    rationale = ["Core documents: Commercial Invoice (CI), Packing List (PL)."]

    # Apply rules
    if not rules_df.empty:
        for _, r in rules_df.iterrows():
            try:
                cond = r["condition_json"]; adds = r["add_forms"]
                if cond_true(cond, payload):
                    required.update(adds)
                    rationale.append(f"Rule matched: {r['name']}")
            except Exception as e:
                st.warning(f"Skipped a bad rule row: {getattr(r,'name','(no name)')}")
    else:
        st.info("No active rules loaded — showing only core forms (CI, PL).")

    req_list = sorted(required)

    # ---- Results
    st.subheader("✅ Required forms")
    st.write(req_list)

    st.subheader("🧾 Fields to capture")
    show_fields = fields_df[fields_df["form_code"].isin(req_list)].copy()
    if show_fields.empty:
        st.warning("No matching fields found. Check that your Fields sheet 'form_code' values match the form codes above.")
    else:
        st.dataframe(show_fields, use_container_width=True)

    st.subheader("📌 Rationale")
    for r in rationale:
        st.write("-", r)

    # ---- Optional PDF generation (only if pdf_utils.py exists)
    if PDF_OK:
        st.subheader("📄 Generate PDFs (demo)")
        # Minimal dummy CI line item for demo
        items_ci = [{
            "description": "Example line",
            "hs_code": hs,
            "qty": 1,
            "unit": "pcs",
            "unit_price": f"{value:,.2f}",
            "amount": f"{value:,.2f}",
        }]
        subtot = value; freight = 0.0; insurance = 0.0; total = subtot + freight + insurance
        base_data = {
            "exporter_name": exporter_name,
            "consignee_name": consignee_name,
            "origin": origin,
            "destination": destination,
            "incoterm": incoterm,
            "currency": currency,
            "invoice_no": invoice_no,
            "invoice_date": "",
            "subtotal": f"{subtot:,.2f}",
            "freight": f"{freight:,.2f}",
            "insurance": f"{insurance:,.2f}",
            "total": f"{total:,.2f}",
            "packages_count": 1,
            "gross_weight_kg": "",
            "net_weight_kg": "",
            "shipment_ref": invoice_no,
        }

        if "CI" in req_list:
            ci_buf = make_ci_pdf(base_data, items_ci)
            st.download_button(
                "⬇️ Download Commercial Invoice (PDF)",
                data=ci_buf.getvalue(),
                file_name=f"{invoice_no}_Commercial_Invoice.pdf",
                mime="application/pdf"
            )
        if "PL" in req_list:
            pl_rows = [{"pkg_no": 1, "description": "Example line", "qty": 1, "unit": "pcs", "gross_wt": "", "net_wt": ""}]
            pl_buf = make_pl_pdf(base_data, pl_rows)
            st.download_button(
                "⬇️ Download Packing List (PDF)",
                data=pl_buf.getvalue(),
                file_name=f"{invoice_no}_Packing_List.pdf",
                mime="application/pdf"
            )

        placeholder_map = {
            "ISPM15_Statement": "ISPM-15 Wood Packaging Statement",
            "CE_DoC": "EU Declaration of Conformity (Placeholder)",
            "RoHS_Declaration": "RoHS Declaration (Placeholder)",
            "REACH_Declaration": "REACH Declaration (Placeholder)",
            "Generic_CoO": "Certificate of Origin (Placeholder)",
            "EU_SAD": "EU SAD Cover Sheet (Reference Only)"
        }
        for code, title in placeholder_map.items():
            if code in req_list:
                lines = [
                    f"{title}", "",
                    f"Exporter: {exporter_name}",
                    f"Consignee: {consignee_name}",
                    f"Origin: {origin}  Destination: {destination}",
                    f"HS (example): {hs}",
                    "",
                    "Demo placeholder PDF generated for prototype purposes.",
                    "Replace with official template mapping in production."
                ]
                buf = make_simple_statement(title, lines)
                st.download_button(
                    f"⬇️ Download {title} (PDF)",
                    data=buf.getvalue(),
                    file_name=f"{code}.pdf",
                    mime="application/pdf"
                )
    else:
        st.info("PDF generation disabled (pdf_utils.py not found). Add it to enable CI/PL PDFs.")
