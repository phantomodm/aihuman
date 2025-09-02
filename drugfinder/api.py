from fastapi import FastAPI, Query
from pydantic import BaseModel
from novabrain_rag import query_index, build_index
from pathlib import Path
from datetime import datetime
from apscheduler.schedulers.background import BackgroundScheduler
from apscheduler.triggers.cron import CronTrigger
from apscheduler.triggers.interval import IntervalTrigger
from remote_config_loader import load_remote_config
import atexit
import subprocess
import json

app = FastAPI(title="NovaBrain API")

# Index directories
INDEX_DIR = Path("indexes")
INDEX_DIR.mkdir(exist_ok=True)

# Separate index files
MEDICAL_INDEX = INDEX_DIR / "medical.index"
GENOMICS_INDEX = INDEX_DIR / "genomics.index"
ALL_INDEX = INDEX_DIR / "all.index"

# Default merged outputs (from nova_fusion)
today_str = datetime.now().strftime("%Y-%m-%d")
DEFAULT_MEDICAL_FILE = Path(f"nova_fusion_medical_{today_str}.jsonl")
DEFAULT_GENOMICS_FILE = Path(f"nova_fusion_genomics_{today_str}.jsonl")
DEFAULT_ALL_FILE = Path(f"nova_fusion_all_{today_str}.jsonl")

class QueryRequest(BaseModel):
    question: str
    top_k: int = 5
    source: str = "medical"  # medical, genomics, all


# --- Scheduler Jobs ---
def run_nova_fusion_phase2():
    """Run Nova Fusion Phase 2 ingestion (PubMed + WHO + Clinical)."""
    print("🕒 Scheduled job: Running Nova Fusion Phase 2...")
    subprocess.run(["python", "nova_fusion.py", "--build-index"])

def run_nova_fusion_genomics():
    """Run Nova Fusion Genomics ingestion (gnomAD, dbSNP, ClinVar, GWAS)."""
    print("🕒 Scheduled job: Running Nova Fusion Genomics...")
    subprocess.run(["python", "nova_fusion_genomics.py", "--build-index"])

# --- Setup APScheduler ---
scheduler = BackgroundScheduler()

# Run Phase 2 every 6 hours
scheduler.add_job(run_nova_fusion_phase2, IntervalTrigger(hours=6), id="phase2")

# Run Genomics ingestion nightly at 2 AM
scheduler.add_job(run_nova_fusion_genomics, CronTrigger(hour=2, minute=0), id="genomics")

# Start scheduler
scheduler.start()
atexit.register(lambda: scheduler.shutdown())


# ---------------------------
# Core Query Endpoints
# ---------------------------

@app.post("/query")
async def query_data(request: QueryRequest):
    """Query one of the indexes: medical, genomics, or all"""
    if request.source == "genomics":
        index_path = GENOMICS_INDEX
        data_file = DEFAULT_GENOMICS_FILE
    elif request.source == "all":
        index_path = ALL_INDEX
        data_file = DEFAULT_ALL_FILE
    else:
        index_path = MEDICAL_INDEX
        data_file = DEFAULT_MEDICAL_FILE

    if not index_path.exists():
        return {"error": f"Index for {request.source} not found. Please build it first."}

    results = query_index(index_path=str(index_path), question=request.question, top_k=request.top_k)
    return {"source": request.source, "results": results}


@app.post("/build-index")
async def build_indexes(medical: bool = True, genomics: bool = False, all_data: bool = False):
    """Build one or more indexes from the latest nova_fusion output files"""
    built = []
    if medical and DEFAULT_MEDICAL_FILE.exists():
        build_index(input_file=DEFAULT_MEDICAL_FILE, output_index=str(MEDICAL_INDEX))
        built.append("medical")
    if genomics and DEFAULT_GENOMICS_FILE.exists():
        build_index(input_file=DEFAULT_GENOMICS_FILE, output_index=str(GENOMICS_INDEX))
        built.append("genomics")
    if all_data and DEFAULT_ALL_FILE.exists():
        build_index(input_file=DEFAULT_ALL_FILE, output_index=str(ALL_INDEX))
        built.append("all")
    return {"built_indexes": built}


@app.get("/status")
async def status():
    """Check which indexes are available"""
    return {
        "medical_index": MEDICAL_INDEX.exists(),
        "genomics_index": GENOMICS_INDEX.exists(),
        "all_index": ALL_INDEX.exists(),
    }


# ---------------------------
# Analytics Endpoints
# ---------------------------

@app.get("/analytics/pubmed/stats")
async def pubmed_stats():
    """Quick summary of PubMed ingestion"""
    if not DEFAULT_MEDICAL_FILE.exists():
        return {"error": "No PubMed/medical file available."}

    articles = []
    with open(DEFAULT_MEDICAL_FILE, "r", encoding="utf-8") as f:
        for line in f:
            rec = json.loads(line)
            if "pmid" in rec:
                articles.append(rec)

    return {
        "total_articles": len(articles),
        "sample_pmids": [a["pmid"] for a in articles[:5]],
    }


@app.get("/analytics/who/indicators")
async def who_indicators_summary(limit: int = 10):
    """Summarize WHO indicators"""
    if not DEFAULT_MEDICAL_FILE.exists():
        return {"error": "No WHO data file available."}

    indicators = {}
    with open(DEFAULT_MEDICAL_FILE, "r", encoding="utf-8") as f:
        for line in f:
            rec = json.loads(line)
            if rec.get("source") == "WHO_GHO":
                ind = rec.get("indicator")
                indicators[ind] = indicators.get(ind, 0) + 1

    sorted_inds = sorted(indicators.items(), key=lambda x: x[1], reverse=True)
    return {"top_indicators": sorted_inds[:limit]}


@app.get("/analytics/genomics/top-variants")
async def top_variants(limit: int = 10):
    """Return most common variant IDs across genomics feeds"""
    if not DEFAULT_GENOMICS_FILE.exists():
        return {"error": "No genomics file available."}

    variants = {}
    with open(DEFAULT_GENOMICS_FILE, "r", encoding="utf-8") as f:
        for line in f:
            rec = json.loads(line)
            if "variant_id" in rec:
                vid = rec["variant_id"]
                variants[vid] = variants.get(vid, 0) + 1

    sorted_vars = sorted(variants.items(), key=lambda x: x[1], reverse=True)
    return {"top_variants": sorted_vars[:limit]}

@app.get("/analytics/overview")
async def analytics_overview():
    """
    Combined snapshot of PubMed, WHO, and Genomics ingestion.
    Useful for dashboards, summaries, and investor reports.
    """

    overview = {
        "pubmed": {},
        "who": {},
        "genomics": {}
    }

    # --- PubMed ---
    if DEFAULT_MEDICAL_FILE.exists():
        pubmed_articles = 0
        sample_pmids = []
        with open(DEFAULT_MEDICAL_FILE, "r", encoding="utf-8") as f:
            for line in f:
                rec = json.loads(line)
                if "pmid" in rec:
                    pubmed_articles += 1
                    if len(sample_pmids) < 5:
                        sample_pmids.append(rec["pmid"])
        overview["pubmed"] = {
            "total_articles": pubmed_articles,
            "sample_pmids": sample_pmids
        }
    else:
        overview["pubmed"] = {"error": "No PubMed data available."}

    # --- WHO ---
    if DEFAULT_MEDICAL_FILE.exists():
        indicators = {}
        with open(DEFAULT_MEDICAL_FILE, "r", encoding="utf-8") as f:
            for line in f:
                rec = json.loads(line)
                if rec.get("source") == "WHO_GHO":
                    ind = rec.get("indicator")
                    indicators[ind] = indicators.get(ind, 0) + 1
        sorted_inds = sorted(indicators.items(), key=lambda x: x[1], reverse=True)
        overview["who"] = {
            "top_indicators": sorted_inds[:5],
            "total_indicators": len(indicators)
        }
    else:
        overview["who"] = {"error": "No WHO data available."}

    # --- Genomics ---
    if DEFAULT_GENOMICS_FILE.exists():
        variants = {}
        with open(DEFAULT_GENOMICS_FILE, "r", encoding="utf-8") as f:
            for line in f:
                rec = json.loads(line)
                if "variant_id" in rec:
                    vid = rec["variant_id"]
                    variants[vid] = variants.get(vid, 0) + 1
        sorted_vars = sorted(variants.items(), key=lambda x: x[1], reverse=True)
        overview["genomics"] = {
            "top_variants": sorted_vars[:5],
            "total_variants": len(variants)
        }
    else:
        overview["genomics"] = {"error": "No genomics data available."}

    return overview

from apscheduler.schedulers.background import BackgroundScheduler
import atexit
import subprocess

# --- Scheduler Jobs ---
def run_nova_fusion_phase2():
    """Run Nova Fusion Phase 2 ingestion (PubMed + WHO + Clinical)."""
    print("🕒 Scheduled job: Running Nova Fusion Phase 2...")
    subprocess.run(["python", "nova_fusion.py", "--build-index"])

def run_nova_fusion_genomics():
    """Run Nova Fusion Genomics ingestion (gnomAD, dbSNP, ClinVar, GWAS)."""
    print("🕒 Scheduled job: Running Nova Fusion Genomics...")
    subprocess.run(["python", "nova_fusion_genomics.py", "--build-index"])

# --- Setup APScheduler ---
scheduler = BackgroundScheduler()

# ✅ Load Firebase Remote Config
config = load_remote_config()
print(f"📡 Loaded scheduler config from Firebase: {config}")

# --- Phase 2 job config ---
phase2_hours = int(config.get("phase2_interval_hours", 6))
scheduler.add_job(
    run_nova_fusion_phase2,
    IntervalTrigger(hours=phase2_hours),
    id="phase2"
)

# --- Genomics job config ---
genomics_hour = int(config.get("genomics_cron_hour", 2))
genomics_minute = int(config.get("genomics_cron_minute", 0))
scheduler.add_job(
    run_nova_fusion_genomics,
    CronTrigger(hour=genomics_hour, minute=genomics_minute),
    id="genomics"
)

# Start scheduler
scheduler.start()
atexit.register(lambda: scheduler.shutdown())

# --- API Endpoints for Scheduler ---
@app.get("/scheduler/jobs")
def list_jobs():
    """List all scheduled jobs with next run times."""
    jobs_info = []
    for job in scheduler.get_jobs():
        jobs_info.append({
            "id": job.id,
            "next_run": str(job.next_run_time),
            "trigger": str(job.trigger)
        })
    return {"scheduled_jobs": jobs_info}

@app.post("/scheduler/run-now")
def run_now(job: str):
    """Trigger a scheduled job manually (e.g., ?job=genomics)."""
    if job == "phase2":
        run_nova_fusion_phase2()
    elif job == "genomics":
        run_nova_fusion_genomics()
    else:
        return {"error": f"Unknown job '{job}'. Valid: phase2, genomics"}
    return {"status": f"Triggered {job} job"}

