#!/usr/bin/env python
"""
ingest_dbsnp.py
Fetch variant metadata from dbSNP (via NCBI E-utilities or JSON REST).
Scheduled feed for large-scale ingestion.
"""

import requests
import json
from pathlib import Path
from datetime import datetime

BASE_URL = "https://api.ncbi.nlm.nih.gov/variation/v0/refsnp"

def fetch_dbsnp(rs_ids, max_results=50):
    results = []
    for rsid in rs_ids[:max_results]:
        url = f"{BASE_URL}/{rsid}"
        resp = requests.get(url)
        if resp.status_code != 200:
            continue
        data = resp.json()
        results.append({
            "source": "dbSNP",
            "rsid": rsid,
            "chrom": data.get("primary_snapshot_data", {}).get("placements_with_allele", [{}])[0].get("seq_id", ""),
            "alleles": data.get("primary_snapshot_data", {}).get("allele_annotations", []),
            "metadata": data
        })
    return results

def main(config=None, query=None, max_results=50, email=None, output_file=None,
         backend="faiss", build_index=False):
    rs_ids = config.get("dbsnp_rsid_list", ["rs7412", "rs429358"]) if config else ["rs7412"]
    results = fetch_dbsnp(rs_ids, max_results=max_results)

    if output_file is None:
        today = datetime.now().strftime("%Y-%m-%d")
        output_file = Path(f"ingest_dbsnp_output_{today}.jsonl")

    with open(output_file, "w", encoding="utf-8") as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(f"💾 Saved {len(results)} dbSNP records to {output_file}")

    if build_index:
        from novabrain_rag import build_index
        build_index(input_file=output_file)
        print("✅ dbSNP indexed.")
