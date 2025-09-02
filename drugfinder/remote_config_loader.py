import os
import firebase_admin
from firebase_admin import credentials, remote_config

_firebase_initialized = False

def load_remote_config():
    """
    Loads Firebase Remote Config parameters into a Python dictionary.
    This is shared by all ingestion scripts so config is centralized.
    """
    global _firebase_initialized

    cred_path = os.getenv("FIREBASE_CRED_JSON", "firebase_service_account.json")
    if not os.path.exists(cred_path):
        raise FileNotFoundError(f"Firebase service account file not found: {cred_path}")

    if not _firebase_initialized:
        cred = credentials.Certificate(cred_path)
        firebase_admin.initialize_app(cred)
        _firebase_initialized = True

    # Pull template from Firebase
    template = remote_config.get_template()
    params = template.parameters

    # Convert Firebase parameter objects into plain dictionary values
    config = {key: val.default_value.value for key, val in params.items() if val.default_value}

    return config

# {
#   "phase2_interval_hours": "8",
#   "genomics_cron_hour": "4",
#   "genomics_cron_minute": "0"
# }
