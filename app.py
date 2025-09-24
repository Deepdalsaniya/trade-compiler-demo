import json, re
import pandas as pd
import streamlit as st

RULES_URL = st.secrets.get("RULES_CSV_URL", "rules.csv")
FIELDS_URL = st.secrets.get("FIELDS_CSV_URL", "fields.csv")

def _clean_json_text(s: str) -> str:
    if not isinstance(s, str): return ""
    # replace smart quotes and stray single quotes
    s = s.replace("“", '"').replace("”", '"').replace("’", "'")
    # sometimes sheets add whitespace/newlines
    return s.strip()

def _safe_read_csv(url_or_path: str, local_fallback: str):
    try:
        return pd.read_csv(url_or_path)
    except Exception as e_remote:
        try:
            st.warning(f"Couldn’t fetch remote CSV, using local fallback: {local_fallback}")
            return pd.read_csv(local_fallback)
        except Exception as e_local:
            st.error("Failed to load both remote and local CSV.")
            st.exception(e_remote)
            st.exception(e_local)
            # return empty df with expected columns so UI still renders
            return pd.DataFrame()

@st.cache_data(ttl=300)
def load_rules():
    df = _safe_read_csv(RULES_URL, "rules.csv")
    if df.empty:
        # ensure columns exist
        df = pd.DataFrame(columns=["name","priority","active","condition_json","add_forms"])
    # normalize columns and strip whitespace
    df.columns = [c.strip() for c in df.columns]
    for c in ["name","priority","active","condition_json","add_forms"]:
        if c not in df.columns: df[c] = None

    # sanitize JSON columns
    df["condition_json"] = df["condition_json"].map(lambda v: _clean_json_text(v))
    df["add_forms"] = df["add_forms"].map(lambda v: _clean_json_text(v))

    # parse JSON with guard
    def _parse_obj(txt, default):
        try:
            return json.loads(txt) if isinstance(txt, str) and txt else default
        except Exception:
            return default

    df["priority"] = pd.to_numeric(df["priority"], errors="coerce").fillna(100).astype(int)
    df["active"]   = pd.to_numeric(df["active"], errors="coerce").fillna(1).astype(int)
    df["condition_json"] = df["condition_json"].apply(lambda s: _parse_obj(s, {}))
    df["add_forms"]      = df["add_forms"].apply(lambda s: _parse_obj(s, []))

    # filter to active but don’t crash if all inactive
    df = df[df["active"] == 1].copy()
    return df.sort_values(["priority"]).reset_index(drop=True)

@st.cache_data(ttl=300)
def load_fields():
    df = _safe_read_csv(FIELDS_URL, "fields.csv")
    if df.empty:
        df = pd.DataFrame(columns=["form_code","field_key","label","type","required"])
    df.columns = [c.strip() for c in df.columns]
    for c in ["form_code","field_key","label","type","required"]:
        if c not in df.columns: df[c] = None
    df["required"] = pd.to_numeric(df["required"], errors="coerce").fillna(1).astype(int)
    return df

st.button("🔄 Reload rules/fields", on_click=lambda: st.cache_data.clear())

