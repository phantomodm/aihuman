#!/usr/bin/env python
"""
ingest_who.py
Ingest WHO Global Health Observatory (GHO) and GHED data.
Compatible with nova_fusion auto-detection and merging.
Uses who_indicators.json for controlled sequential fetch.
"""

import requests
import json
from pathlib import Path
from datetime import datetime

WHO_GHO_BASE = "https://ghoapi.azureedge.net/api"
WHO_GHED_BASE = "https://apps.who.int/nha/database"
CONFIG_FILE = Path("who_indicators.json")
RAW_DIR = Path("data/raw")
RAW_DIR.mkdir(parents=True, exist_ok=True)

def fetch_gho_indicators(indicator_list=None, max_results=None):
    """Fetch WHO GHO indicators sequentially."""
    data = []

    if not indicator_list:
        # fallback: pull all available indicators
        resp = requests.get(f"{WHO_GHO_BASE}/Indicator")
        resp.raise_for_status()
        indicator_list = [item["IndicatorCode"] for item in resp.json()["value"]]

    for indicator in indicator_list:
        url = f"{WHO_GHO_BASE}/{indicator}"
        print(f"📡 Fetching WHO GHO indicator: {indicator}")
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
    """Stub for WHO GHED (Health Expenditure) data."""
    return [{
        "source": "WHO_GHED",
        "indicator": "HealthExpenditurePerCapita",
        "country": "GLOBAL",
        "year": "2021",
        "value": None,
        "metadata": {"note": "TODO: implement full GHED CSV ingestion"}
    }]

def main(config=None, query=None, max_results=100, email=None, output_file=None, backend="faiss", build_index=False):
    # ✅ Load indicator list from config file
    if CONFIG_FILE.exists():
        with open(CONFIG_FILE, "r", encoding="utf-8") as f:
            config_data = json.load(f)
            indicators = config_data.get("who_indicators", [])
    else:
        print("⚠️ who_indicators.json not found → defaulting to ALL indicators")
        indicators = None

    print(f"🌍 Fetching WHO GHO indicators...")
    gho_data = fetch_gho_indicators(indicator_list=indicators, max_results=max_results)

    print(f"💰 Fetching WHO GHED (stub)...")
    ghed_data = fetch_ghed(max_results=max_results)

    all_data = gho_data + ghed_data
    print(f"✅ Retrieved {len(all_data)} WHO records.")

    if output_file is None:
        today_str = datetime.now().strftime("%Y-%m-%d")
        output_file = RAW_DIR / f"ingest_who_raw_{today_str}.jsonl"

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
