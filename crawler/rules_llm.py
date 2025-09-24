# crawler/rules_llm.py
import json, sqlite3, pandas as pd

def export_rules(db="crawler.db", out_csv="rules_generated.csv"):
    con = sqlite3.connect(db)
    rows = con.execute("SELECT payload FROM extractions WHERE kind='rules'").fetchall()
    con.close()
    merged = []
    for (payload,) in rows:
        try:
            for r in json.loads(payload):
                merged.append({
                    "name": r.get("notes","LLM rule"),
                    "priority": 90,
                    "active": 0,  # start inactive -> human review
                    "condition_json": json.dumps(r.get("condition_json", {})),
                    "add_forms": json.dumps(r.get("add_forms", [])),
                })
        except: pass
    if not merged: return
    pd.DataFrame(merged).to_csv(out_csv, index=False)
    print("Exported", len(merged), "rules →", out_csv)

