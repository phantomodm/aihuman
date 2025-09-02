import json
from pathlib import Path
from datetime import datetime
import pydicom
from remote_config_loader import load_remote_config

def extract_dicom_metadata(file_path: Path) -> dict:
    """Extract selected metadata fields from DICOM header."""
    ds = pydicom.dcmread(str(file_path), stop_before_pixels=True)
    return {
        "sop_instance_uid": getattr(ds, "SOPInstanceUID", ""),
        "study_date": getattr(ds, "StudyDate", ""),
        "modality": getattr(ds, "Modality", ""),
        "body_part": getattr(ds, "BodyPartExamined", ""),
        "institution": getattr(ds, "InstitutionName", ""),
        "manufacturer": getattr(ds, "Manufacturer", ""),
        "series_description": getattr(ds, "SeriesDescription", ""),
    }

def main(config=None, query=None, max_results=50, email=None,
         output_file=None, backend="faiss", build_index=False):
    """
    Standard Nova Fusion ingestion signature for DICOM.
    Writes raw metadata JSONL into output dir. De-ID handled by nova_fusion.
    """
    config = config or load_remote_config()
    dicom_dir = Path(config.get("dicom_input_dir", "dicom_samples"))
    output_dir = Path(config.get("dicom_output_dir", "data/raw"))
    output_dir.mkdir(parents=True, exist_ok=True)

    today = datetime.now().strftime("%Y-%m-%d")
    if output_file is None:
        output_file = output_dir / f"ingest_dicom_output_{today}.jsonl"

    dicom_files = list(dicom_dir.glob("*.dcm"))[:max_results]
    print(f"🖼 Found {len(dicom_files)} DICOM files to ingest")

    with open(output_file, "w", encoding="utf-8") as f:
        for dicom_file in dicom_files:
            record = extract_dicom_metadata(dicom_file)
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    print(f"✅ Saved DICOM metadata to {output_file}")

    if build_index:
        from novabrain_rag import build_index
        print("🧠 Building vector index from DICOM data...")
        build_index(input_file=output_file)
        print("✅ DICOM data indexed.")

    return output_file  # 🔑 so nova_fusion knows where the raw file is

if __name__ == "__main__":
    main()
