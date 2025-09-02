# main.py
import functions_framework
from google.cloud import speech_v1p1beta1 as speech
import firebase_admin
from firebase_admin import firestore
from datetime import datetime
import json
import os

# --- Global Scope: Load resources once on cold start ---
client = speech.SpeechClient()
firebase_admin.initialize_app()
db = firestore.client()

SEMANTIC_SEARCH_URL = os.getenv('SEMANTIC_SEARCH_URL')

# --- Global Scope: Load resources once on cold start ---
client = speech.SpeechClient()
firebase_admin.initialize_app()
db = firestore.client()

@functions_framework.http
def speech_to_text(request):
    """
    Receives an audio file, transcribes it, and saves the transcription to Firestore.
    """
    # Get the data from the FormData object sent by the front-end
    if request.method != 'POST':
        return {'error': 'Method not supported.'}, 405

    # Get the audio content and sessionId from the request
    session_id = request.form.get('sessionId')
    audio_content = request.files.get('audio').read()
    language_code = request.form.get('language') or 'en-US'
    
    if not session_id or not audio_content:
        return {'error': 'sessionId and audio data are required.'}, 400

    # 1. Configure the STT API call with the correct language
    audio = speech.RecognitionAudio(content=audio_content)
    config = speech.RecognitionConfig(
        encoding=speech.RecognitionConfig.AudioEncoding.LINEAR16,
        sample_rate_hertz=16000,
        language_code=language_code,
    )
    
    # 2. Call the STT API
    try:
        response = client.recognize(config=config, audio=audio)
        transcribed_text = response.results[0].alternatives[0].transcript
        print(f"Transcribed Text: {transcribed_text}")
    except Exception as e:
        print(f"Error during transcription: {e}")
        return {'error': 'Error transcribing audio.'}, 500

    # 3. Create a ConversationObject for the user's message
    user_message_object = {
        'id': '',
        'sender': 'user',
        'sessionId': session_id,
        'text': transcribed_text,
        'timestamp': datetime.now().isoformat(),
        'results': [],
        'read': True,
    }
    
    # 4. Save the object to Firestore, which will trigger the AIBrain™
    doc_ref = db.collection('conversations').add(user_message_object)
    
    # Update the document with its own ID
    doc_ref.update({'id': doc_ref.id})

    # Return a success message. The front-end is listening for the AI's response.
    return {'status': 'transcription_successful'}, 200

# gcloud functions deploy speech_to_text \
# --runtime python39 \
# --entry-point speech_to_text \
# --trigger-http