import json
from pathlib import Path
from datetime import datetime

RAW_DIR = Path("data/raw")
RAW_DIR.mkdir(parents=True, exist_ok=True)

def main():
    today = datetime.now().strftime("%Y-%m-%d")
    output_file = RAW_DIR / f"ingest_labs_raw_{today}.jsonl"

    # Example lab results
    lab_results = [
        {"patient_id": "PID123", "test": "Hemoglobin", "value": "13.2", "unit": "g/dL", "date": "2024-01-01"},
        {"patient_id": "PID456", "test": "WBC", "value": "7.1", "unit": "10^9/L", "date": "2024-01-02"},
    ]

    with open(output_file, "w", encoding="utf-8") as f:
        for result in lab_results:
            f.write(json.dumps(result, ensure_ascii=False) + "\n")

    print(f"✅ Saved lab results to {output_file}")

if __name__ == "__main__":
    main()
