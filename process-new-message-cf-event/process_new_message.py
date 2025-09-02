# main.py
import functions_framework
import numpy as np
import torch
import os
import tempfile
import json
import datetime
import hashlib
from vertexai.preview.language_models import TextGenerationModel
from google.cloud import texttospeech_v1 as tts
from transformers import AutoTokenizer, AutoModel
from pinecone import init, Index
import firebase_admin
from firebase_admin import firestore
from google.cloud import storage
from google.cloud import texttospeech_v1 as tts

# --- Global Scope: Load resources once on cold start ---
# Load the embedding model and tokenizer
tokenizer = AutoTokenizer.from_pretrained('sentence-transformers/all-MiniLM-L6-v2')
model = AutoModel.from_pretrained('sentence-transformers/all-MiniLM-L6-v2')
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

# Initialize Pinecone client
init(api_key=os.getenv('PINECONE_API_KEY'), environment=os.getenv('PINECONE_ENVIRONMENT'))
pinecone_index = Index('aihuman-core')

# Initialize Firebase Admin SDK for Firestore
firebase_admin.initialize_app()
db = firestore.client()

storage_client = storage.Client()
tts_client = tts.TextToSpeechClient()
llm_model = TextGenerationModel.from_pretrained("text-bison")

def get_vector(text_list):
    """
    Generates a list of vector embeddings for a given list of text strings.
    """
    # Tokenize the text, convert to PyTorch tensors, and move to the device
    encoded_input = tokenizer(text_list, padding=True, truncation=True, return_tensors='pt').to(device)

    # Generate the embeddings with the model
    with torch.no_grad():
        model_output = model(**encoded_input)

    # Take the mean of all token embeddings to get a single vector per sentence
    sentence_embeddings = model_output.last_hidden_state.mean(dim=1)

    # Return the vectors as a NumPy array on the CPU
    return sentence_embeddings.cpu().numpy()

@functions_framework.event
def process_new_message(cloud_event):
    """
    This function is triggered by a new document being added to the Firestore
    'conversations' collection.
    """
    message_data = cloud_event.data['value']['fields']

    # Only process messages from the 'user'
    if message_data['sender']['stringValue'] != 'user':
        return
    
    user_text = message_data['text']['stringValue']
    session_id = message_data['sessionId']['stringValue']
    doc_id = message_data['id']['stringValue']

    #query firestor for previous messages in the same session
    conversation_ref = db.collection('conversations')
    query = conversation_ref.where('sessionId', '==', session_id).order_by('timestamp')
    docs = query.stream()
    

    # request_json = request.get_json(silent=True)
    # user_symptom_text = request_json.get('symptoms')
    # session_id = request_json.get('sessionId')
    
    # if not user_symptom_text or not session_id:
    #     return {'error': 'Symptoms and sessionId are required.'}, 400

    # --- Check the cache first ---
    cache_key = hashlib.sha256(user_text.encode('utf-8')).hexdigest()
    cache_ref = db.collection('search_cache').document(cache_key)
    cached_result = cache_ref.get()

    if cached_result.exists:
        print("Cache hit! Returning cached result.")
        results_list = cached_result.to_dict()['results']
        ai_response_text = cached_result.to_dict()['result']
    else:
        conversation_history = get_conversation_history_from_firestore(session_id)
        print("Cache miss. Performing new search.")
        
        # --- Perform the semantic search ---
        query_vector = get_vector([user_text])
        pinecone_response = pinecone_index.query(
            vector=query_vector.tolist()[0],
            top_k=3,
            include_metadata=True
        )

        results_list = [
            {
                "text": match['metadata']['original_text'],
                "confidence_score": match['score'],
                "metadata": match['metadata']
            }
            for match in pinecone_response['matches']
        ]

        ai_response_text = generate_natural_response(results_list[0]['text'], conversation_history)

        # --- Store the result in the cache ---
        cache_ref.set({'results': results_list, 'result': ai_response_text})

    # --- Create and save the AI's ConversationObject to Firestore ---
    conversation_object = {
        'id': None,
        'sender': 'ai',
        'sessionId': session_id,
        'text': ai_response_text,
        'timestamp': datetime.now().isoformat(),
        'results': results_list,
        'audio_url': None,
        'video_url': None
    }
    
    # We now save this to a subcollection of the session
    doc_ref = db.collection('conversations').document()
    conversation_object['id'] = doc_ref.id  # Set the document ID

    # --- Synthesize the AI's response to audio and save it to Cloud Storage ---
    audio_url = synthesize_and_save_audio(ai_response_text, doc_id)
    conversation_object['audio_url'] = audio_url

    doc_ref.set(conversation_object)  # Update the document with the audio URL

    # In our real-time architecture, the front-end listens to Firestore.
    # We just need to return a success message.
    return {'status': 'success'}, 200

def synthesize_and_save_audio(text, message_id):
    """
    Synthesizes audio from text and saves it to Cloud Storage.
    """
    voice = tts.VoiceSelectionParams(
        language_code="en-US",
        name="en-US-Wavenet-F",
        ssml_gender=tts.SsmlVoiceGender.FEMALE,
    )
    audio_config = tts.AudioConfig(
        audio_encoding=tts.AudioEncoding.MP3
    )

    synthesis_input = tts.SynthesisInput(text=text)
    tts_response = tts_client.synthesize_speech(
        input=synthesis_input, voice=voice, audio_config=audio_config
    )

    # Save the audio file to Cloud Storage
    bucket_name = os.getenv('AUDIO_BUCKET_NAME')
    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(f'{message_id}.mp3')
    blob.upload_from_string(tts_response.audio_content, content_type='audio/mpeg')
    
    return blob.public_url

def generate_natural_response(raw_text):
    if raw_text:
        prompt = f"""
        Act as a compassionate and ethical AI doctor. 
        Take the following medical information and rephrase it into a natural, easy-to-understand response for a patient.
        If the information is incomplete or a definitive diagnosis, use a disclaimer.
        
        Information: '{raw_text}'
        """
        response = llm_model.predict(prompt=prompt, temperature=0.2, max_output_tokens=1024)
        return response.text
    
    return "I'm sorry, I couldn't find a relevant answer for that."

# A helper function to get the conversation history from Firestore
def get_conversation_history_from_firestore(session_id):
    messages_ref = db.collection('conversations').document(session_id).collection('messages')
    query = messages_ref.order_by('timestamp', direction=firestore.Query.ASCENDING)
    docs = query.stream()
    
    conversation_history = ""
    for doc in docs:
        message = doc.to_dict()
        conversation_history += f"{message['sender']}: {message['text']}\n"
    
    return conversation_history


# def generate_natural_responseb(results_list):
#     """
#     (MVP Approach) Generates a natural-sounding response based on the top result.
#     """
#     if results_list:
#         # Get the original text from the best match
#         raw_text = results_list[0]['text']
        
#         # This is a simple, rule-based rewrite.
#         return f"Based on our analysis, we found some information related to your query: '{raw_text}'. Do you have any further questions?"

# def generate_natural_response_a(results_list):
#     """
#     (Long-Term Vision) Uses a generative AI model to create a natural-sounding response.
#     """
#     if results_list:
#         raw_text = results_list[0]['text']
#         model = TextGenerationModel.from_pretrained("text-bison")
        
#         # This is the key part: crafting a prompt that instructs the AI
#         prompt = f"""
#         Act as a compassionate and ethical AI doctor. 
#         Take the following medical information and rephrase it into a natural, easy-to-understand response for a patient.
#         If the information is incomplete or a definitive diagnosis, use a disclaimer.
        
#         Information: '{raw_text}'
#         """
        
#         # The model generates the final, human-friendly text
#         response = model.predict(prompt=prompt, temperature=0.2, max_output_tokens=1024)
#         return response.text
    
#     return "I'm sorry, I couldn't find a relevant answer for that."

# gcloud functions deploy process_new_message --runtime python39 --trigger-event providers/cloud.firestore/eventTypes/document.create --trigger-resource 'projects/[PROJECT_ID]/databases/(default)/documents/conversations/{sessionId}/messages/{messageId}'