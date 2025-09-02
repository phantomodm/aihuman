import requests
import json
from pathlib import Path
from datetime import datetime
from novabrain_rag import build_index

# --- Fetch PubMed articles ---
def fetch_pubmed(query, email, max_results=50):
    base_url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"
    params = {
        "db": "pubmed",
        "term": query,
        "retmax": max_results,
        "retmode": "json",
        "email": email
    }
    search_resp = requests.get(base_url, params=params)
    search_resp.raise_for_status()
    ids = search_resp.json()["esearchresult"]["idlist"]

    if not ids:
        return []

    fetch_url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"
    fetch_params = {
        "db": "pubmed",
        "id": ",".join(ids),
        "retmode": "xml"
    }
    fetch_resp = requests.get(fetch_url, params=fetch_params)
    fetch_resp.raise_for_status()

    return [{"pmid": pmid, "raw_xml": fetch_resp.text} for pmid in ids]

# --- Standard main ---
def main(config=None, query=None, max_results=50, email=None, output_file=None, backend="faiss", build_index=False):
    # ✅ Pull values from config first, fall back to CLI args
    query = query or (config.get("pubmed_query") if config else None)
    email = email or (config.get("pubmed_email") if config else None)
    max_results = max_results or int(config.get("pubmed_max_results", 50) if config else 50)

    if not query or not email:
        raise ValueError("PubMed ingestion requires query and email (either CLI args or config).")

    print(f"📚 Fetching PubMed articles for query='{query}', max_results={max_results}")

    results = fetch_pubmed(query, email, max_results)

    if output_file is None:
        today_str = datetime.now().strftime("%Y-%m-%d")
        output_file = Path(f"ingest_pubmed_output_{today_str}.jsonl")

    with open(output_file, "w", encoding="utf-8") as f:
        for rec in results:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    print(f"💾 Saved {len(results)} PubMed records to {output_file}")

    if build_index:
        print("🧠 Building vector index from PubMed data...")
        build_index(input_file=output_file)
        print("✅ PubMed data indexed.")
