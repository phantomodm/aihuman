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
import base64

# --- Global Scope: Load resources once ---
tokenizer = AutoTokenizer.from_pretrained('sentence-transformers/all-MiniLM-L6-v2')
model = AutoModel.from_pretrained('sentence-transformers/all-MiniLM-L6-v2')
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

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

def get_relevant_info(query_text):
    """
    Performs a semantic search and returns the top results.
    """
    query_vector = get_vector([query_text])
    pinecone_response = pinecone_index.query(
        vector=query_vector.tolist()[0],
        top_k=5, # Get more results for a richer article
        include_metadata=True
    )
    return [match['metadata']['original_text'] for match in pinecone_response['matches']]

def generate_article_text(topic, relevant_info):
    """
    Uses a generative AI model to create an article based on a topic and context.
    """
    context_text = "\n".join(relevant_info)
    prompt = f"""
    Act as a professional health writer for AIHumanMedia.
    Write a short, informative, and easy-to-understand article for the public about '{topic}'.
    Use the following medical information as your source of truth. Do not make up any information.
    
    Source Information:
    {context_text}
    
    Article:
    """
    response = llm_model.predict(prompt=prompt, temperature=0.2, max_output_tokens=2048)
    return response.text

@functions_framework.cloud_event
def generate_article(cloud_event):
    """
    Triggered by a Pub/Sub message.
    """
    # Get the data from the Pub/Sub message
    if 'message' in cloud_event.data:
        message_data = base64.b64decode(cloud_event.data['message']['data']).decode('utf-8')
        message_json = json.loads(message_data)
        topic = message_json.get('topic')
    else:
        return {'error': 'No message in payload.'}, 400

    if not topic:
        return {'error': 'Topic is required.'}, 400

    # 1. Perform a semantic search to get relevant information
    relevant_info = get_relevant_info(topic)

    # 2. Use the generative AI to create the article
    article_text = generate_article_text(topic, relevant_info)

    # 3. Save the article to Firestore for review
    db.collection('aihuman_media_articles').add({
        'topic': topic,
        'article': article_text,
        'status': 'pending_review',
        'timestamp': datetime.now().isoformat()
    })

    print(f"Article for '{topic}' successfully generated and saved for review.")
    return 'OK', 200