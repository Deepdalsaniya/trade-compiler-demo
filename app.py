import json
import re
from pathlib import Path

import pandas as pd
import streamlit as st

st.set_page_config(page_title="Compliance Compiler (MVP)", page_icon="🌍")

@st.cache_data
def load_rules():
    df = pd.read_csv("rules.csv")
    # normalize
    df["priority"] = df["priority"].fillna(100).astype(int)
    df["active"] = df["active"].fillna(1).astype(int)
    # parse JSON-ish columns
    def j(v):
        try:
            return json.loads(v)
        except Exception:
            return {}
    def jarr(v):
        try:
            x = json.loads(v)
            return x if isinstance(x, list) else []
        except Exception:
            return []
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
        prefixes.append(parts[0])             # '85'
    if len(parts) >= 2:
        prefixes.append(parts[0] + parts[1])  # '8517'
        prefixes.append(parts[0] + "." + parts[1])  # '85.17'
    if len(parts) >= 3:
        prefixes.append(parts[0] + parts[1] + parts[2])       # '851762'
        prefixes.append(parts[0] + "." + parts[1] + "." + parts[2])  # '85.17.62'
        prefixes.append(parts[0] + parts[1] + "." + parts[2])        # '8517.62'
    # unique order
    seen = set()
    out = []
    for p in prefixes:
        if p not in seen:
            out.append(p); seen.add(p)
    return out

def cond_true(cond: dict, payload: dict) -> bool:
    # Supported keys: origin, destination_in, hs_prefix_in, value_gt, packaging_eq, flags_has
    if "origin" in cond and payload.get("origin") != cond["origin"]:
        return False
    if "destination_in" in cond and payload.get("destination") not in cond["destination_in"]:
        return False
    if "hs_prefix_in" in cond:
        prefs = payload.get("hs_prefixes", [])
        targets = cond["hs_prefix_in"]
        ok = False
        for t in targets:
            if any(p.startswith(t) or p == t for p in prefs):
                ok = True; break
        if not ok:
            return False
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
st.caption("No code needed. Enter shipment details → get required forms + fields. Download the JSON bundle.")

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
        cond = r["condition_json"]; adds = r["add_forms"]
        if cond_true(cond, payload):
            required.update(adds)
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

    # JSON packet to download
    packet = {
        "input": {k: payload[k] for k in ["origin","destination","hs","value","packaging","flags"]},
        "required_forms": req_list,
        "fields": show_fields.to_dict(orient="records"),
        "rationale": rationale,
    }
    st.download_button("⬇️ Download JSON bundle", data=json.dumps(packet, indent=2),
                       file_name="compliance_packet.json", mime="application/json")
