# crawler/extract.py
import json, time, sqlite3, os
import pandas as pd
import openai

openai.api_key = os.getenv("OPENAI_API_KEY")

PROMPT = """You are a trade-compliance analyst.
From the document text below, extract *actionable requirements* as JSON objects with keys:
{ "form_code": string, "condition_json": object, "add_forms": [string], "notes": string }.
Use our form codes when applicable: CI, PL, SLI, EEI_Worksheet, EU_SAD, Generic_CoO, CE_DoC, RoHS_Declaration, REACH_Declaration, ISPM15_Statement.
Only output a JSON array.
TEXT:
"""

def llm_extract(text):
    if not openai.api_key:
        return []
    try:
        r = openai.ChatCompletion.create(
            model="gpt-4o-mini",
            messages=[{"role":"system","content":"Be precise and concise."},
                      {"role":"user","content":PROMPT + text[:6000]}],
            temperature=0.2
        )
        return json.loads(r.choices[0].message["content"])
    except Exception:
        return []

def run(db="crawler.db"):
    con = sqlite3.connect(db)
    docs = pd.read_sql_query("SELECT url, text_excerpt FROM documents WHERE status=1", con)
    for _, row in docs.iterrows():
        rules = llm_extract(row["text_excerpt"] or "")
        if not rules: continue
        con.execute("INSERT INTO extractions(url, extracted_at, kind, payload) VALUES(?,?,?,?)",
                    (row["url"], int(time.time()), "rules", json.dumps(rules)))
    con.commit(); con.close()

if __name__ == "__main__":
    run()

