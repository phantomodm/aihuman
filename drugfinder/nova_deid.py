import re
import json
from pathlib import Path
from datetime import datetime, timedelta

RAW_DIR = Path("data/raw")
DEID_DIR = Path("data/deid")
DEID_DIR.mkdir(parents=True, exist_ok=True)


def scrub_text(text: str) -> str:
    """Regex scrub for PHI-like patterns."""
    patterns = [
        r"\b\d{3}-\d{2}-\d{4}\b",   # SSN
        r"\b\d{10}\b",              # Phone numbers
        r"[A-Z][a-z]+ [A-Z][a-z]+", # Names (rough)
        r"\d{1,2}/\d{1,2}/\d{2,4}"  # Dates
    ]
    for p in patterns:
        text = re.sub(p, "[REDACTED]", text)
    return text


def shift_date(date_str: str) -> str:
    """Shift date by deterministic offset to break re-ID risk."""
    try:
        dt = datetime.fromisoformat(date_str)
        offset = timedelta(days=round((hash(date_str) % 365) - 180))
        return (dt + offset).strftime("%Y-%m-%d")
    except Exception:
        return "[REDACTED_DATE]"


def deid_record(record: dict) -> dict:
    """Recursive de-identification of record fields."""
    rec = {}
    for k, v in record.items():
        if isinstance(v, str):
            if "date" in k.lower():
                rec[k] = shift_date(v)
            else:
                rec[k] = scrub_text(v)
        elif isinstance(v, list):
            rec[k] = [scrub_text(str(x)) for x in v]
        elif isinstance(v, dict):
            rec[k] = deid_record(v)
        else:
            rec[k] = v
    return rec


def process_file(input_file: Path) -> Path:
    """De-identify one file and return path to de-id output."""
    output_file = DEID_DIR / input_file.name.replace("output", "deid")
    with open(input_file, "r", encoding="utf-8") as fin, \
         open(output_file, "w", encoding="utf-8") as fout:
        for line in fin:
            record = json.loads(line)
            clean = deid_record(record)
            fout.write(json.dumps(clean, ensure_ascii=False) + "\n")
    print(f"✅ De-identified {input_file} → {output_file}")
    return output_file


def main():
    for f in RAW_DIR.glob("*.jsonl"):
        process_file(f)


if __name__ == "__main__":
    main()
