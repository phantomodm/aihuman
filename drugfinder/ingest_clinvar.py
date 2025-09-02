#!/usr/bin/env python
"""
ingest_clinvar.py
Fetch variant significance & clinical assertions from ClinVar.
"""

import requests
import json
from pathlib import Path
from datetime import datetime

BASE_URL = "https://api.ncbi.nlm.nih.gov/variation/v0/clinvar"

def fetch_clinvar(query="BRCA1", max_results=50):
    url = f"{BASE_URL}?q={query}&rows={max_results}"
    resp = requests.get(url)
    resp.raise_for_status()
    data = resp.json()
    results = []
    for rec in data.get("data", []):
        results.append({
            "source": "ClinVar",
            "id": rec.get("variation_id"),
            "gene": rec.get("gene", ""),
            "clinical_significance": rec.get("clinical_significance", ""),
            "conditions": rec.get("conditions", []),
            "metadata": rec
        })
    return results

def main(config=None, query=None, max_results=50, email=None, output_file=None,
         backend="faiss", build_index=False):
    query = query or (config.get("clinvar_query", "BRCA1") if config else "BRCA1")
    results = fetch_clinvar(query=query, max_results=max_results)

    if output_file is None:
        today = datetime.now().strftime("%Y-%m-%d")
        output_file = Path(f"ingest_clinvar_output_{today}.jsonl")

    with open(output_file, "w", encoding="utf-8") as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(f"💾 Saved {len(results)} ClinVar records to {output_file}")

    if build_index:
        from novabrain_rag import build_index
        build_index(input_file=output_file)
        print("✅ ClinVar indexed.")
