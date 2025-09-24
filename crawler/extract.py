# crawler/extract.py

# Import necessary Python libraries
import json, time, sqlite3, os
import pandas as pd
import openai

# Set the OpenAI API key from your environment variable
openai.api_key = os.getenv("OPENAI_API_KEY")

# This is the prompt sent to the LLM to get requirements from a document text.
PROMPT = """You are a trade-compliance analyst.
From the document text below, extract *actionable requirements* as JSON objects with keys:
{ "form_code": string, "condition_json": object, "add_forms": [string], "notes": string }.
Use our form codes when applicable: CI, PL, SLI, EEI_Worksheet, EU_SAD, Generic_CoO, CE_DoC, RoHS_Declaration, REACH_Declaration, ISPM15_Statement.
Only output a JSON array.
TEXT:
"""

# This function asks the LLM (GPT) to extract requirements from text
def llm_extract(text):
    # If there is no API key, just return an empty list (skip extraction)
    if not openai.api_key:
        return []
    try:
        # Call the OpenAI ChatCompletion API with the prompt and text
        r = openai.ChatCompletion.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "Be precise and concise."},  # Tell the model to keep responses short and clear
                {"role": "user", "content": PROMPT + text[:6000]}          # Add the prompt and text (limit to 6000 chars)
            ],
            temperature=0.2  # Lower temperature for more consistent output
        )
        # Parse the model's output as JSON and return it
        return json.loads(r.choices[0].message["content"])
    except Exception:
        # If anything goes wrong (e.g., API error), return an empty list
        return []

# This function runs the extraction process for all documents in the database
def run(db="crawler.db"):
    # Connect to the SQLite database (default: crawler.db)
    con = sqlite3.connect(db)
    # Read all documents that have status=1 (ready for extraction) into a pandas DataFrame
    docs = pd.read_sql_query("SELECT url, text_excerpt FROM documents WHERE status=1", con)
    # Go through each document
    for _, row in docs.iterrows():
        # Use the LLM to extract rules from each document's text excerpt
        rules = llm_extract(row["text_excerpt"] or "")
        # If no rules were found, skip to the next document
        if not rules:
            continue
        # Insert the extracted rules into the 'extractions' table in the database
        con.execute(
            "INSERT INTO extractions(url, extracted_at, kind, payload) VALUES(?,?,?,?)",
            (row["url"], int(time.time()), "rules", json.dumps(rules))
        )
    # Save (commit) all database changes and close the connection
    con.commit()
    con.close()

# If this script is run directly (not imported), start the extraction process
if __name__ == "__main__":
    run()
