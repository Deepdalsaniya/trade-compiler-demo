import json, re
import pandas as pd
import streamlit as st

# Read from secrets if present; otherwise fall back to local CSVs
RULES_URL = st.secrets.get("RULES_CSV_URL", "rules.csv")
FIELDS_URL = st.secrets.get("FIELDS_CSV_URL", "fields.csv")

st.set_page_config(page_title="Compliance Compiler (MVP)", page_icon="🌍")

def _parse_json(v, default_obj):
    try:
        j = json.loads(v)
        return j if isinstance(j, type(default_obj)) or default_obj is None else default_obj
    except Exception:
        return default_obj

@st.cache_data(ttl=300)  # cache 5 minutes; adjust as you like
def load_rules():
    try:
        df = pd.read_csv(RULES_URL)
    except Exception:
        df = pd.read_csv("rules.csv")  # local fallback
    df["priority"] = pd.to_numeric(df.get("priority", 100), errors="coerce").fillna(100).astype(int)
    df["active"] = pd.to_numeric(df.get("active", 1), errors="coerce").fillna(1).astype(int)
    df["condition_json"] = df["condition_json"].apply(lambda v: _parse_json(v, {}))
    df["add_forms"] = df["add_forms"].apply(lambda v: _parse_json(v, []))
    return df.sort_values(["priority"]).reset_index(drop=True)

@st.cache_data(ttl=300)
def load_fields():
    try:
        df = pd.read_csv(FIELDS_URL)
    except Exception:
        df = pd.read_csv("fields.csv")  # local fallback
    df["required"] = pd.to_numeric(df.get("required", 1), errors="coerce").fillna(1).astype(int)
    return df

# Add a reload button somewhere near the top of your page:
st.button("🔄 Reload rules/fields", on_click=lambda: (st.cache_data.clear()))
