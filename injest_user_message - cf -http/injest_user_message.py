import functions_framework
from datetime import datetime
import firebase_admin
from firebase_admin import firestore

firebase_admin.initialize_app()
db = firestore.client()


@functions_framework.http
def ingest_user_text(request):
    # Extract user message and session ID from the request
    request_json = request.get_json(silent=True)
    user_text = request_json.get('text')
    session_id = request_json.get('sessionId')

    if not user_text or not session_id:
        return {'error': 'Message and sessionId are required.'}, 400
    user_collection = {
        'id':'',
        'sender': 'user',
        'text': user_text,
        'sessionId': session_id,
        'timestamp': datetime.now().isoformat(),
        'results': [],
        'audio_url': None,
        'video_url': None,
        'read': False
    }

    # Save the user message to Firestore
    docRef = db.collection('conversations').add(user_collection)
    new_id = docRef.id  # Get the document ID
    docRef.update({'id': new_id})

    return {'status': 'success'}, 200
