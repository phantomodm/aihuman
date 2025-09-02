import json
from pathlib import Path
from datetime import datetime
import requests
from novabrain_config import load_remote_config

def fetch_fhir_resources(base_url, resource_type, auth_token=None, max_results=100):
    """Fetch FHIR resources from endpoint with pagination."""
    headers = {"Accept": "application/fhir+json"}
    if auth_token:
        headers["Authorization"] = f"Bearer {auth_token}"

    url = f"{base_url}/{resource_type}?_count={max_results}"
    all_results = []

    while url:
        print(f"📡 Fetching {resource_type} from {url}")
        resp = requests.get(url, headers=headers)
        resp.raise_for_status()
        data = resp.json()

        if "entry" in data:
            for entry in data["entry"]:
                all_results.append(entry["resource"])

        # Handle pagination (FHIR "next" link)
        url = None
        if "link" in data:
            for link in data["link"]:
                if link.get("relation") == "next":
                    url = link["url"]

    return all_results

def main(config=None):
    """Standard ingestion for FHIR (EHR data)."""
    config = config or load_remote_config()
    print(f"📡 Using config from Firebase: {config}")

    base_url = config.get("fhir_base_url")
    auth_token = config.get("fhir_auth_token", None)
    resources = config.get("fhir_resources", ["Patient", "Condition", "Observation", "MedicationRequest"])
    max_results = int(config.get("fhir_max_results", "100"))

    if not base_url:
        raise ValueError("❌ Missing 'fhir_base_url' in config.")

    all_data = []
    for res in resources:
        res_data = fetch_fhir_resources(base_url, res, auth_token, max_results=max_results)
        print(f"✅ Retrieved {len(res_data)} {res} records")
        all_data.extend(res_data)

    today_str = datetime.now().strftime("%Y-%m-%d")
    output_file = Path(f"ingest_fhir_output_{today_str}.jsonl")

    with open(output_file, "w", encoding="utf-8") as f:
        for rec in all_data:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    print(f"💾 Saved {len(all_data)} FHIR records to {output_file}")

    # Optional vectorization
    if config.get("build_index", False):
        from novabrain_rag import build_index
        print("🧠 Building vector index from FHIR data...")
        build_index(input_file=output_file)
        print("✅ FHIR data indexed.")

if __name__ == "__main__":
    main()
