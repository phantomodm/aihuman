import requests
import json
from pathlib import Path
from datetime import datetime
from novabrain_rag import build_index

# --- Fetch ClinicalTrials.gov studies ---
def fetch_clinicaltrials(max_results=200):
    url = "https://clinicaltrials.gov/api/v2/studies"
    params = {
        "format": "json",
        "pageSize": max_results,
        "fields": ",".join([
            "protocolSection.identificationModule.nctId",
            "protocolSection.conditionsModule.conditions",
            "protocolSection.armsInterventionsModule.interventions",
            "protocolSection.identificationModule.briefTitle",
            "protocolSection.descriptionModule.briefSummary",
            "protocolSection.statusModule.overallStatus",
            "protocolSection.designModule.phases",
            "protocolSection.sponsorCollaboratorsModule.leadSponsor.name",
            "protocolSection.statusModule.startDateStruct.date",
            "protocolSection.statusModule.completionDateStruct.date"
        ])
    }

    resp = requests.get(url, params=params)
    resp.raise_for_status()
    return resp.json().get("studies", [])

# --- Standard main ---
def main(config=None, query=None, max_results=200, email=None, output_file=None, backend="faiss", build_index=False):
    # ✅ Pull values from config first, fall back to CLI args
    max_results = max_results or int(config.get("clinicaltrials_max_results", 200) if config else 200)

    print(f"🧪 Fetching up to {max_results} ClinicalTrials.gov records...")

    results = fetch_clinicaltrials(max_results)

    if output_file is None:
        today_str = datetime.now().strftime("%Y-%m-%d")
        output_file = Path(f"ingest_clinicaltrials_output_{today_str}.jsonl")

    with open(output_file, "w", encoding="utf-8") as f:
        for rec in results:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    print(f"💾 Saved {len(results)} ClinicalTrials records to {output_file}")

    if build_index:
        print("🧠 Building vector index from ClinicalTrials data...")
        build_index(input_file=output_file)
        print("✅ ClinicalTrials data indexed.")
