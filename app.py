import json
import re
from pathlib import Path
import pandas as pd
import streamlit as st

from pdf_utils import make_ci_pdf, make_pl_pdf, make_simple_statement  # NEW: PDF

st.set_page_config(page_title="Compliance Compiler (MVP)", page_icon="🌍")

@st.cache_data
def load_rules():
    df = pd.read_csv("rules.csv")
    df["priority"] = df["priority"].fillna(100).astype(int)
    df["active"] = df["active"].fillna(1).astype(int)
    def j(v):
        try: return json.loads(v)
        except: return {}
    def jarr(v):
        try:
            x = json.loads(v)
            return x if isinstance(x, list) else []
        except: return []
    df["condition_json"] = df["condition_json"].apply(j)
    df["add_forms"] = df["add_forms"].apply(jarr)
    return df.sort_values(["priority"]).reset_index(drop=True)

@st.cache_data
def load_fields():
    df = pd.read_csv("fields.csv")
    df["required"] = df["required"].fillna(1).astype(int)
    return df

def hs_prefixes(hs: str):
    hs = (hs or "").replace(" ", "")
    parts = re.split(r"[.\-]", hs)
    prefixes = []
    if len(parts) >= 1 and parts[0]:
        prefixes.append(parts[0])
    if len(parts) >= 2:
        prefixes.append(parts[0] + parts[1])
        prefixes.append(parts[0] + "." + parts[1])
    if len(parts) >= 3:
        prefixes.append(parts[0] + parts[1] + parts[2])
        prefixes.append(parts[0] + "." + parts[1] + "." + parts[2])
        prefixes.append(parts[0] + parts[1] + "." + parts[2])
    # unique
    out, seen = [], set()
    for p in prefixes:
        if p not in seen: out.append(p); seen.add(p)
    return out

def cond_true(cond: dict, payload: dict) -> bool:
    if "origin" in cond and payload.get("origin") != cond["origin"]:
        return False
    if "destination_in" in cond and payload.get("destination") not in cond["destination_in"]:
        return False
    if "hs_prefix_in" in cond:
        prefs = payload.get("hs_prefixes", [])
        targets = cond["hs_prefix_in"]
        ok = any(any(p.startswith(t) or p == t for p in prefs) for t in targets)
        if not ok: return False
    if "value_gt" in cond:
        v = payload.get("value")
        if v is None or float(v) <= float(cond["value_gt"]):
            return False
    if "packaging_eq" in cond and payload.get("packaging") != cond["packaging_eq"]:
        return False
    if "flags_has" in cond and cond["flags_has"] not in payload.get("flags", []):
        return False
    return True

rules_df = load_rules()
fields_df = load_fields()

st.title("🌍 Compliance Compiler (US → EU Demo)")
st.caption("Enter shipment → get required forms + fields. Now with PDF generation (CI & PL).")

with st.expander("Shipment details", expanded=True):
    col1, col2, col3 = st.columns(3)
    with col1:
        origin = st.selectbox("Origin", ["US"])
        hs = st.text_input("HS Code", "8517.62")
        incoterm = st.text_input("Incoterm", "DAP")
    with col2:
        destination = st.selectbox("Destination", ["DE", "EU"])
        value = st.number_input("Invoice Value (USD)", min_value=0.0, value=12000.0, step=100.0)
        currency = st.text_input("Currency", "USD")
    with col3:
        packaging = st.selectbox("Packaging", ["standard", "wood"])
        flags = st.multiselect("Flags", ["controlled"])

with st.expander("Parties & shipment meta", expanded=True):
    col1, col2, col3 = st.columns(3)
    with col1:
        exporter_name = st.text_input("Exporter Name", "Acme Exporters LLC")
        invoice_no = st.text_input("Invoice No", "INV-001")
    with col2:
        consignee_name = st.text_input("Consignee Name", "Beta GmbH")
        invoice_date = st.date_input("Invoice Date")
    with col3:
        shipment_ref = st.text_input("Shipment Ref", "REF-001")
        eori = st.text_input("Importer EORI (for EU_SAD)", "")

with st.expander("Line items (Commercial Invoice / Packing List)", expanded=True):
    st.caption("Edit directly; amounts auto-calc in demo when you click Generate PDFs.")
    items_df = st.data_editor(
        pd.DataFrame([
            {"description":"4G Modem","hs_code":hs,"qty":10,"unit":"pcs","unit_price":120.00},
            {"description":"Ethernet Cable","hs_code":"8544.42","qty":20,"unit":"pcs","unit_price":5.50},
        ]),
        num_rows="dynamic",
        use_container_width=True
    )

with st.expander("Weights & packages (Packing List)", expanded=True):
    col1, col2, col3 = st.columns(3)
    with col1:
        packages_count = st.number_input("Packages Count", min_value=1, value=2, step=1)
    with col2:
        gross_weight_kg = st.number_input("Gross Weight (kg)", min_value=0.0, value=25.0, step=0.1)
    with col3:
        net_weight_kg = st.number_input("Net Weight (kg)", min_value=0.0, value=22.0, step=0.1)

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

    required = {"CI", "PL"}  # core forms always
    rationale = ["Core documents: Commercial Invoice (CI), Packing List (PL)."]

    for _, r in rules_df.iterrows():
        if r["active"] != 1:
            continue
        if cond_true(r["condition_json"], payload):
            required.update(r["add_forms"])
            rationale.append(f"Rule matched: {r['name']}")

    req_list = sorted(required)
    st.subheader("✅ Required forms")
    st.write(req_list)

    st.subheader("🧾 Fields to capture")
    show_fields = fields_df[fields_df["form_code"].isin(req_list)].copy()
    if show_fields.empty:
        st.info("No field metadata yet. (Add rows to fields.csv)")
    else:
        st.dataframe(show_fields, use_container_width=True)

    # --- NEW: PDF generation controls ---
    st.subheader("📄 Generate PDFs (demo)")
    # compute invoice amounts
    items_ci = []
    subtotal = 0.0
    for _, r in items_df.iterrows():
        qty = float(r.get("qty",0) or 0)
        price = float(r.get("unit_price",0) or 0)
        amt = qty * price
        subtotal += amt
        items_ci.append({
            "description": r.get("description",""),
            "hs_code": r.get("hs_code",""),
            "qty": qty,
            "unit": r.get("unit",""),
            "unit_price": f"{price:,.2f}",
            "amount": f"{amt:,.2f}",
        })
    freight = 0.00
    insurance = 0.00
    total = subtotal + freight + insurance

    base_data = {
        "exporter_name": exporter_name,
        "consignee_name": consignee_name,
        "origin": origin,
        "destination": destination,
        "incoterm": incoterm,
        "currency": currency,
        "invoice_no": invoice_no,
        "invoice_date": str(invoice_date),
        "subtotal": f"{subtotal:,.2f}",
        "freight": f"{freight:,.2f}",
        "insurance": f"{insurance:,.2f}",
        "total": f"{total:,.2f}",
        "packages_count": int(packages_count),
        "gross_weight_kg": gross_weight_kg,
        "net_weight_kg": net_weight_kg,
        "shipment_ref": shipment_ref,
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
        # derive simple packing rows
        pl_rows = []
        pkg_no = 1
        for _, r in items_df.iterrows():
            pl_rows.append({
                "pkg_no": pkg_no,
                "description": r.get("description",""),
                "qty": r.get("qty",""),
                "unit": r.get("unit",""),
                "gross_wt": "",  # leave per-package wt empty in demo
                "net_wt": "",
            })
            pkg_no += 1
        pl_buf = make_pl_pdf(base_data, pl_rows)
        st.download_button(
            "⬇️ Download Packing List (PDF)",
            data=pl_buf.getvalue(),
            file_name=f"{shipment_ref}_Packing_List.pdf",
            mime="application/pdf"
        )

    # Quick placeholder one-pagers for other forms in the required list
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
                f"{title}",
                "",
                f"Exporter: {exporter_name}",
                f"Consignee: {consignee_name}",
                f"Origin: {origin}  Destination: {destination}",
                f"HS (example): {hs}",
                "",
                "This is a demo placeholder PDF generated for prototype purposes.",
                "Replace with official template mapping in production."
            ]
            buf = make_simple_statement(title, lines)
            st.download_button(
                f"⬇️ Download {title} (PDF)",
                data=buf.getvalue(),
                file_name=f"{code}.pdf",
                mime="application/pdf"
            )

    # JSON packet download (unchanged)
    packet = {
        "input": {k: v for k, v in dict(
            origin=origin, destination=destination, hs=hs, value=value,
            packaging=packaging, flags=flags, incoterm=incoterm, currency=currency,
            exporter_name=exporter_name, consignee_name=consignee_name,
            invoice_no=invoice_no, invoice_date=str(invoice_date),
            shipment_ref=shipment_ref,
            packages_count=int(packages_count),
            gross_weight_kg=gross_weight_kg,
            net_weight_kg=net_weight_kg
        ).items()},
        "required_forms": req_list,
        "fields": show_fields.to_dict(orient="records"),
        "rationale": rationale,
        "items": items_ci
    }
    st.download_button("⬇️ Download JSON bundle", data=json.dumps(packet, indent=2),
                       file_name="compliance_packet.json", mime="application/json")
