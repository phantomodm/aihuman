import argparse
import importlib
import pkgutil
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
import json
from datetime import datetime
from remote_config_loader import load_remote_config
from nova_deid import process_file

# Ingestion package
INGEST_PACKAGE = "ingestion_feeds"

def discover_feeds():
    """Discover ingestion feed scripts dynamically."""
    feeds = []
    package = importlib.import_module(INGEST_PACKAGE)
    for _, module_name, _ in pkgutil.iter_modules(package.__path__):
        if module_name.startswith("ingest_"):
            feeds.append(f"{INGEST_PACKAGE}.{module_name}")
    return feeds

def run_feed(feed_module, config, args, output_file=None):
    """Run a feed ingestion script with standardized main()."""
    try:
        module = importlib.import_module(feed_module)
        if hasattr(module, "main"):
            print(f"▶ Running {feed_module}.main()")
            module.main(
                config=config,
                query=args.query,
                max_results=args.max_results,
                email=args.email,
                output_file=output_file,
                backend=args.backend,
                build_index=False
            )
        else:
            print(f"⚠ {feed_module} has no main()")
    except Exception as e:
        print(f"❌ Error running {feed_module}: {e}")

def merge_outputs(output_files, merged_file):
    """Merge multiple JSONL outputs into one file."""
    with open(merged_file, "w", encoding="utf-8") as fout:
        for file in output_files:
            with open(file, "r", encoding="utf-8") as fin:
                for line in fin:
                    fout.write(line)

def main():
    parser = argparse.ArgumentParser(description="Nova Fusion Ingestion")
    parser.add_argument("--query", help="Optional query param (e.g. gene, condition)")
    parser.add_argument("--max-results", type=int, default=200, help="Max results per feed")
    parser.add_argument("--email", help="Email (for PubMed, etc.)")
    parser.add_argument("--backend", choices=["faiss", "pinecone", "weaviate"], default="faiss")
    parser.add_argument("--build-index", action="store_true", help="Build vector index after ingestion")
    parser.add_argument("--config", help="Optional JSON config file for feeds", default=None)
    parser.add_argument("--include-genomics", action="store_true", help="Include genomics feeds (slower, sequential)")
    args = parser.parse_args()

    # Load config once
    config = load_remote_config()
    print(f"📡 Loaded Firebase Remote Config: {config}")

    # Discover all feeds
    feeds = discover_feeds()
    print(f"🔍 Found ingestion scripts: {feeds}")

    # Split feeds by type
    genomic_suffixes = ("gnomad", "1000genomes", "dbsnp", "clinvar", "gwas")
    medical_feeds = [f for f in feeds if not f.endswith(genomic_suffixes)]
    genomic_feeds = [f for f in feeds if f.endswith(genomic_suffixes)]

    today_str = datetime.now().strftime("%Y-%m-%d")
    medical_outputs, genomic_outputs = [], []

    # Run medical feeds in parallel
    print("⚡ Running medical feeds in parallel...")
    with ThreadPoolExecutor(max_workers=len(medical_feeds)) as executor:
        futures = {}
        for feed in medical_feeds:
            output_file = Path(f"{feed.replace('.', '_')}_output_{today_str}.jsonl")
            futures[executor.submit(run_feed, feed, config, args, output_file)] = (feed, output_file)

        for future in as_completed(futures):
            feed, raw_file = futures[future]
            try:
                future.result()
                requires_deid = config.get(f"{feed}_requires_deid", False)
                if requires_deid:
                    print(f"🔒 Applying de-identification for {feed}")
                    result_file = process_file(raw_file)
                else:
                    result_file = raw_file
                medical_outputs.append(result_file)
            except Exception as exc:
                print(f"❌ {feed} failed: {exc}")

    # Run genomic feeds sequentially
    if args.include_genomics:
        print("🧬 Running genomics feeds sequentially...")
        for feed in genomic_feeds:
            output_file = Path(f"{feed.replace('.', '_')}_output_{today_str}.jsonl")
            run_feed(feed, config, args, output_file)
            genomic_outputs.append(output_file)

    # Merge separately
    medical_merged = Path(f"nova_fusion_medical_{today_str}.jsonl")
    if medical_outputs:
        merge_outputs(medical_outputs, medical_merged)
        print(f"✅ Merged {len(medical_outputs)} medical outputs → {medical_merged}")

    genomic_merged = None
    if genomic_outputs:
        genomic_merged = Path(f"nova_fusion_genomics_{today_str}.jsonl")
        merge_outputs(genomic_outputs, genomic_merged)
        print(f"✅ Merged {len(genomic_outputs)} genomic outputs → {genomic_merged}")

    # Optionally merge everything into one master file
    merged_file = Path(f"nova_fusion_all_{today_str}.jsonl")
    merge_outputs([f for f in [medical_merged, genomic_merged] if f], merged_file)
    print(f"🌐 Merged all feeds into {merged_file}")

    # Build index from medical only (safer for now)
    if args.build_index:
        from novabrain_rag import build_index
        print("🧠 Building vector index from medical feeds only...")
        build_index(input_file=medical_merged)
        print("✅ Medical vector index built.")

if __name__ == "__main__":
    main()
