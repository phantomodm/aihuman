#!/usr/bin/env python
"""
ingest_gwas.py
Fetch GWAS Catalog associations via EBI API.
"""

import requests
import json
from pathlib import Path
from datetime import datetime

BASE_URL = "https://www.ebi.ac.uk/gwas/rest/api/associations"

def fetch_gwas(query="diabetes", max_results=50):
    url = f"{BASE_URL}?diseaseTrait={query}&size={max_results}"
    resp = requests.get(url)
    resp.raise_for_status()
    data = resp.json()
    results = []
    for rec in data.get("_embedded", {}).get("associations", []):
        results.append({
            "source": "GWAS",
            "trait": query,
            "snp": rec.get("loci", [{}])[0].get("rsId", ""),
            "pvalue": rec.get("pvalue", ""),
            "odds_ratio": rec.get("orPerCopyNum", ""),
            "metadata": rec
        })
    return results

def main(config=None, query=None, max_results=50, email=None, output_file=None,
         backend="faiss", build_index=False):
    query = query or (config.get("gwas_trait", "diabetes") if config else "diabetes")
    results = fetch_gwas(query=query, max_results=max_results)

    if output_file is None:
        today = datetime.now().strftime("%Y-%m-%d")
        output_file = Path(f"ingest_gwas_output_{today}.jsonl")

    with open(output_file, "w", encoding="utf-8") as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(f"💾 Saved {len(results)} GWAS records to {output_file}")

    if build_index:
        from novabrain_rag import build_index
        build_index(input_file=output_file)
        print("✅ GWAS indexed.")
