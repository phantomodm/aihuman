import functions_framework
import json
import os
import base64
from google.cloud import bigquery
from google.cloud.firestore_v1.base_query import FieldFilter
from google.cloud.firestore_v1.query import Query

# --- Global Scope: Load resources once ---
bigquery_client = bigquery.Client()
project_id = os.getenv('GOOGLE_CLOUD_PROJECT')
dataset_id = 'your_aihuman_dataset'
table_id = 'aihuman_learning_data'

@functions_framework.cloud_event
def stream_to_bigquery(cloud_event):
    """
    Triggered by a new document in the 'aihuman_learning_data' Firestore collection.
    """
    # Get the data from the Firestore event
    document_data = cloud_event.data['value']['fields']
    
    # 1. Format the data for BigQuery
    # Firestore data is structured differently, so we need to flatten it
    row = {
        "geohash": document_data.get('geohash', {}).get('stringValue'),
        "user_input_text": document_data.get('user_input_text', {}).get('stringValue'),
        "ai_response_text": document_data.get('ai_response_text', {}).get('stringValue'),
        "timestamp": document_data.get('timestamp', {}).get('stringValue'),
        "user_vector": document_data.get('user_vector', {}).get('listValue', {}).get('values'),
        "ai_vector": document_data.get('ai_vector', {}).get('listValue', {}).get('values'),
    }

    # 2. Get the BigQuery table reference
    table_ref = bigquery_client.dataset(dataset_id).table(table_id)
    
    # 3. Stream the data to BigQuery
    errors = bigquery_client.insert_rows_json(table_ref, [row])
    
    if errors:
        print(f"Encountered errors while inserting rows: {errors}")
    else:
        print("Successfully streamed data to BigQuery.")

    return 'OK', 200

# gcloud functions deploy stream_to_bigquery \
# --runtime python39 \
# --entry-point stream_to_bigquery \
# --trigger-event providers/cloud.firestore/eventTypes/document.create \
# --trigger-resource 'projects/your-project-id/databases/(default)/documents/aihuman_learning_data/{documentId}'