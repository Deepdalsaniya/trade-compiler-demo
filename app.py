# app.py — Safe, end-to-end Streamlit app (Sheets + fallback + UI + compile + optional PDFs)

import json, re
import pandas as pd
import streamlit as st

st.set_page_config(page_title="Compliance Compiler (MVP)", page_icon="🌍")

# ---- Optional PDF support (won't break if pdf_utils.py is missing)
try:
    from pdf_utils import make_ci_pdf, make_pl_pdf, make_simple_statement
    PDF_OK = True
except Exception:
    PDF_OK = False

# ---- 1) CONFIG / CONSTANTS
RULES_URL = st.secrets.get("RULES_CSV_URL", "rules.csv")
FIELDS_URL = st.secrets.get("FIELDS_CSV_URL", "fields.csv")

# ---- 2) HELPERS + SAFE CSV LOADER
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

# ---- 8) Minimal input UI
col1, col2, col3 = st.columns(3)
with col1:
    origin = st.selectbox("Origin", ["US"])
    hs = st.text_input("HS Code", "8517.62")
with col2:
    destination = st.selectbox("Destination", ["DE", "EU"])
    value = st.number_input("Invoice Value (USD)", min_value=0.0, value=12000.0, step=100.0)
with col3:
    packaging = st.selectbox("Packaging", ["standard", "wood"])
flags = st.multiselect("Flags", ["controlled"])

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
