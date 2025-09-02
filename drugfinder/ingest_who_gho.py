#!/usr/bin/env python
"""
ingest_who_gho.py
Ingest WHO Global Health Observatory (GHO) and Global Health Expenditure Database (GHED) data.
Compatible with nova_fusion.py auto-detection and merging.
"""

import requests
import json
from pathlib import Path
from datetime import datetime

WHO_GHO_BASE = "https://ghoapi.azureedge.net/api"
WHO_GHED_BASE = "https://apps.who.int/nha/database"

def fetch_gho_indicators(indicator_list=None, max_results=None):
    """
    Fetch WHO GHO indicators.
    If indicator_list is None, will fetch all indicators metadata.
    """
    data = []
    if not indicator_list:
        resp = requests.get(f"{WHO_GHO_BASE}/Indicator")
        resp.raise_for_status()
        indicator_list = [item["IndicatorCode"] for item in resp.json()["value"]]

    for indicator in indicator_list:
        url = f"{WHO_GHO_BASE}/{indicator}"
        resp = requests.get(url)
        resp.raise_for_status()
        results = resp.json().get("value", [])
        for r in results[:max_results] if max_results else results:
            data.append({
                "source": "WHO_GHO",
                "indicator": indicator,
                "country": r.get("SpatialDim"),
                "year": r.get("TimeDim"),
                "value": r.get("Value"),
                "metadata": r
            })
    return data

def fetch_ghed(max_results=None):
    """
    WHO GHED requires download of CSV files.
    For now, we fetch metadata and simulate — can be replaced with full CSV ingestion if desired.
    """
    # Placeholder GHED dataset metadata
    ghed_data = [{
        "source": "WHO_GHED",
        "indicator": "HealthExpenditurePerCapita",
        "country": "GLOBAL",
        "year": "2021",
        "value": None,
        "metadata": {"note": "Full GHED CSV ingestion can be implemented here."}
    }]
    return ghed_data

def main(query=None, max_results=100, email=None, output_file=None, backend="faiss", build_index=False):
    print(f"🌍 Fetching WHO GHO indicators (query ignored, pulling all available)...")
    gho_data = fetch_gho_indicators(max_results=max_results)

    print(f"💰 Fetching WHO GHED health expenditure data...")
    ghed_data = fetch_ghed(max_results=max_results)

    all_data = gho_data + ghed_data
    print(f"✅ Retrieved {len(all_data)} WHO records.")

    if output_file is None:
        today_str = datetime.now().strftime("%Y-%m-%d")
        output_file = Path(f"ingest_who_gho_output_{today_str}.jsonl")

    with open(output_file, "w", encoding="utf-8") as f:
        for rec in all_data:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    print(f"💾 Saved WHO GHO/GHED data to {output_file}")

    if build_index:
        from novabrain_rag import build_index
        print("🧠 Building vector index from WHO data...")
        build_index(input_file=output_file)
        print("✅ WHO data indexed.")

if __name__ == "__main__":
    main()
