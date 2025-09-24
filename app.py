import json, re
import pandas as pd
import streamlit as st

st.set_page_config(page_title="Compliance Compiler (MVP)", page_icon="🌍")

# ---- 1) CONFIG / CONSTANTS (must exist before loaders use them)
RULES_URL = st.secrets.get("RULES_CSV_URL", "rules.csv")
FIELDS_URL = st.secrets.get("FIELDS_CSV_URL", "fields.csv")

# ---- 2) HELPERS + SAFE CSV LOADER
def _clean_json_text(s: str) -> str:
    if not isinstance(s, str): 
        return ""
    return s.replace("“", '"').replace("”", '"').replace("’", "'").strip()

def _safe_read_csv(url_or_path: str, local_fallback: str):
    try:
        return pd.read_csv(url_or_path)
    except Exception as e_remote:
        try:
            st.warning(f"Couldn’t fetch remote CSV, using local fallback: {local_fallback}")
            return pd.read_csv(local_fallback)
        except Exception as e_local:
            st.error("Failed to load both remote and local CSV. Rendering minimal UI.")
            st.exception(e_remote)
            st.exception(e_local)
            return pd.DataFrame()  # keep UI alive

# ---- 3) CACHED LOADERS
@st.cache_data(ttl=300)
def load_rules():
    df = _safe_read_csv(RULES_URL, "rules.csv")
    if df.empty:
        df = pd.DataFrame(columns=["name","priority","active","condition_json","add_forms"])

    df.columns = [c.strip() for c in df.columns]
    for c in ["name","priority","active","condition_json","add_forms"]:
        if c not in df.columns: 
            df[c] = None

    df["condition_json"] = df["condition_json"].map(_clean_json_text)
    df["add_forms"] = df["add_forms"].map(_clean_json_text)

    def _parse_obj(txt, default):
        try:
            return json.loads(txt) if isinstance(txt, str) and txt else default
        except Exception:
            return default

    df["priority"] = pd.to_numeric(df["priority"], errors="coerce").fillna(100).astype(int)
    df["active"]   = pd.to_numeric(df["active"], errors="coerce").fillna(1).astype(int)
    df["condition_json"] = df["condition_json"].apply(lambda s: _parse_obj(s, {}))
    df["add_forms"]      = df["add_forms"].apply(lambda s: _parse_obj(s, []))

    # keep only active rules; still safe if it results in empty
    return df[df["active"] == 1].sort_values(["priority"]).reset_index(drop=True)

@st.cache_data(ttl=300)
def load_fields():
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

# ---- 5) NOW load the dataframes (after functions exist)
try:
    rules_df = load_rules()
    fields_df = load_fields()
except Exception as e:
    st.error("Problem loading rules/fields; using empty dataframes so the page still renders.")
    st.exception(e)
    rules_df = pd.DataFrame(columns=["name","priority","active","condition_json","add_forms"])
    fields_df = pd.DataFrame(columns=["form_code","field_key","label","type","required"])

# ---- 6) Continue with the rest of your app UI below
st.title("🌍 Compliance Compiler (US → EU Demo)")
