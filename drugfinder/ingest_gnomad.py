import requests
import json
from pathlib import Path
from datetime import datetime
from novabrain_rag import build_index

# --- Fetch gnomAD variants (simplified) ---
def fetch_gnomad_variants(chrom="1", start=1000000, stop=1001000, dataset="gnomad_r3", max_results=50):
    url = f"https://gnomad.broadinstitute.org/api"
    query = """
    query Variant($chrom: String!, $start: Int!, $stop: Int!, $dataset: DatasetId!, $limit: Int!) {
      variants(region: {chrom: $chrom, start: $start, stop: $stop}, dataset: $dataset, limit: $limit) {
        variantId
        chrom
        pos
        ref
        alt
        consequence
        gene {
          geneSymbol
        }
      }
    }
    """
    variables = {
        "chrom": chrom,
        "start": start,
        "stop": stop,
        "dataset": dataset,
        "limit": max_results
    }

    resp = requests.post(url, json={"query": query, "variables": variables})
    resp.raise_for_status()
    data = resp.json()
    return data.get("data", {}).get("variants", [])

# --- Standard main ---
def main(config=None, query=None, max_results=50, email=None, output_file=None, backend="faiss", build_index=False):
    # ✅ Pull values from config first, fall back to CLI args
    chrom = None
    start = None
    stop = None

    if config:
        chrom = config.get("gnomad_chromosome", "1")
        start = int(config.get("gnomad_start_position", 1000000))
        stop = int(config.get("gnomad_end_position", 1001000))
        max_results = int(config.get("gnomad_max_results", 50))

    # CLI overrides config if provided
    if query and ":" in query:
        chrom, coords = query.split(":")
        start, stop = map(int, coords.split("-"))

    dataset = config.get("gnomad_dataset", "gnomad_r3") if config else "gnomad_r3"

    print(f"🧬 Fetching gnomAD variants: chrom={chrom}, start={start}, stop={stop}, dataset={dataset}, limit={max_results}")

    results = fetch_gnomad_variants(
        chrom=chrom, start=start, stop=stop, dataset=dataset, max_results=max_results
    )

    if output_file is None:
        today_str = datetime.now().strftime("%Y-%m-%d")
        output_file = Path(f"ingest_gnomad_output_{today_str}.jsonl")

    with open(output_file, "w", encoding="utf-8") as f:
        for rec in results:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    print(f"💾 Saved {len(results)} gnomAD variant records to {output_file}")

    if build_index:
        print("🧠 Building vector index from gnomAD data...")
        build_index(input_file=output_file)
        print("✅ gnomAD data indexed.")
