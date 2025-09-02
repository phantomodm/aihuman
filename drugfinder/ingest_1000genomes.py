#!/usr/bin/env python
"""
ingest_1000genomes.py
Ingest metadata from the 1000 Genomes Project (via Ensembl/EBI API).
This script is designed for scheduled/batch runs (not parallel API pulls).
"""

import requests
import json
from pathlib import Path
from datetime import datetime

RAW_DIR = Path("data/raw")
RAW_DIR.mkdir(parents=True, exist_ok=True)

ENSEMBL_API = "https://rest.ensembl.org"

def fetch_variants(chrom="1", start=1000000, end=1001000, max_results=100):
    """
    Fetch variant data from Ensembl for a given region.
    """
    url = f"{ENSEMBL_API}/overlap/region/human/{chrom}:{start}-{end}?feature=variation"
    headers = {"Content-Type": "application/json"}
    resp = requests.get(url, headers=headers)

    if resp.status_code != 200:
        print(f"❌ Error fetching variants: {resp.status_code}")
        return []

    data = resp.json()
    results = []
    for var in data[:max_results]:
        results.append({
            "source": "1000Genomes",
            "id": var.get("id"),
            "chromosome": chrom,
            "start": var.get("start"),
            "end": var.get("end"),
            "strand": var.get("strand"),
            "consequence": var.get("consequence_type"),
        })
    return results

def main(config=None, query=None, max_results=100, email=None, output_file=None,
         backend="faiss", build_index=False):
    """
    Standardized ingestion main().
    """
    chrom = "1"
    start, end = 1000000, 1001000

    if query and ":" in query:
        chrom, coords = query.split(":")
        start, end = map(int, coords.split("-"))

    print(f"🧬 Fetching 1000 Genomes variants for {chrom}:{start}-{end}")

    results = fetch_variants(chrom, start, end, max_results=max_results)

    if output_file is None:
        today = datetime.now().strftime("%Y-%m-%d")
        output_file = RAW_DIR / f"ingest_1000genomes_raw_{today}.jsonl"

    with open(output_file, "w", encoding="utf-8") as f:
        for rec in results:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    print(f"💾 Saved {len(results)} 1000 Genomes records to {output_file}")

    if build_index:
        from novabrain_rag import build_index
        print("🧠 Building vector index from 1000 Genomes data...")
        build_index(input_file=output_file)
        print("✅ 1000 Genomes data indexed.")

if __name__ == "__main__":
    main()
