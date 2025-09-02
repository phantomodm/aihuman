import functions_framework
import numpy as np
import torch
import os
import firebase_admin
from firebase_admin import firestore
import json
import hashlib
from datetime import datetime
from google.cloud import storage
from google.cloud import texttospeech_v1 as tts
from pinecone import init, Index
from transformers import AutoTokenizer, AutoModel
from vertexai.preview.language_models import TextGenerationModel

# --- Global Scope: Load resources once ---
tokenizer = AutoTokenizer.from_pretrained('sentence-transformers/all-MiniLM-L6-v2')
model = AutoModel.from_pretrained('sentence-transformers/all-MiniLM-L6-v2')
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

# Initialize Pinecone and Firestore clients
init(api_key=os.getenv('PINECONE_API_KEY'), environment=os.getenv('PINECONE_ENVIRONMENT'))
pinecone_index = Index('aihuman-core')

firebase_admin.initialize_app()
db = firestore.client()

storage_client = storage.Client()
tts_client = tts.TextToSpeechClient()
llm_model = TextGenerationModel.from_pretrained("text-bison")


def get_vector(text_list):
    """Generates a vector embedding for a given text string."""
    encoded_input = tokenizer(text_list, padding=True, truncation=True, return_tensors='pt').to(device)
    with torch.no_grad():
        model_output = model(**encoded_input)
    sentence_embeddings = model_output.last_hidden_state.mean(dim=1)
    return sentence_embeddings.cpu().numpy()

@functions_framework.event
def continuous_learning_pipeline(cloud_event):
    # Get the data from the Firestore event
    message_data = cloud_event.data['value']['fields']

    # Get the message ID from the path
    message_id = cloud_event.data['value']['name'].split('/')[-1]

    # We only want to process AI's responses for learning
    if message_data['sender']['stringValue'] != 'ai':
        return
           
    # Get the session ID from the message data fields
    session_id = message_data['sessionId']['stringValue']
        
    ai_response_text = message_data['text']['stringValue']
    user_symptom_text = message_data['request']['stringValue'] # Assuming we save this in metadata
    
    messages_ref = db.collection('conversations')
    docs = messages_ref.where('sessionId', '==', session_id).order_by('timestamp', 'DESCENDING').stream()
    conversation_docs = [doc.to_dict() for doc in docs]

    if len(conversation_docs) < 2:
        print("Not enough messages in the session to process. Skipping.")
        return
    
    # Assuming the two most recent messages are the user's and the AI's
    user_message = next(msg for msg in conversation_docs if msg['sender'] == 'user')
    ai_message = next(msg for msg in conversation_docs if msg['sender'] == 'ai')

    # Vectorize both the user's input and the AI's response
    ai_vector = get_vector([ai_response_text])
    user_vector = get_vector([user_symptom_text])

    # --- Upsert the new data into Pinecone ---
    vectors_to_upsert = [
        (f'{message_id}-user', user_vector[0].tolist(), {'original_text': user_symptom_text, 'sender': 'user', 'sessionId': session_id}),
        (f'{message_id}-ai', ai_vector[0].tolist(), {'original_text': ai_response_text, 'sender': 'ai', 'sessionId': session_id})
    ]

    pinecone_index.upsert(vectors=vectors_to_upsert)
    
    # Create new documents or entries for our learning database
    # This is a key step where we can decide to save to a separate collection
    db.collection('aihuman_learning_data').add({
        'user_input_text': user_symptom_text,
        'ai_response_text': ai_response_text,
        'user_vector': user_vector.tolist(),
        'ai_vector': ai_vector.tolist(),
        'timestamp': datetime.now().isoformat(),
        'geohash': message_data.get('geohash', {}).get('stringValue', None)
    })
    
    print(f"Processed message {message_id} for session {session_id}.")
    return {'status': 'success'}, 200

# gcloud functions deploy continuous_learning_pipeline \
# --runtime python39 \
# --trigger-event providers/cloud.firestore/eventTypes/document.create \
# --trigger-resource 'projects/your-project-id/databases/(default)/documents/conversations/{messageId}'
