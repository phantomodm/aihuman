import json
from pathlib import Path
from datetime import datetime

RAW_DIR = Path("data/raw")
RAW_DIR.mkdir(parents=True, exist_ok=True)

def parse_hl7_message(hl7_str: str) -> dict:
    """Simplified HL7 parser (normally use hl7apy or similar)."""
    fields = hl7_str.split("|")
    return {
        "message_type": fields[0] if fields else "",
        "patient_id": fields[2] if len(fields) > 2 else "",
        "observation": fields[3] if len(fields) > 3 else "",
        "date": fields[4] if len(fields) > 4 else "",
    }

def main():
    today = datetime.now().strftime("%Y-%m-%d")
    output_file = RAW_DIR / f"ingest_hl7_raw_{today}.jsonl"

    # Example HL7 messages
    hl7_messages = [
        "MSH|^~\\&|LAB|HOSP|20240101|PID123|OBX|WBC|2024-01-01",
        "MSH|^~\\&|LAB|HOSP|20240102|PID456|OBX|Hemoglobin|2024-01-02"
    ]

    with open(output_file, "w", encoding="utf-8") as f:
        for msg in hl7_messages:
            record = parse_hl7_message(msg)
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    print(f"✅ Saved HL7 messages to {output_file}")

if __name__ == "__main__":
    main()
