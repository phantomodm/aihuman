# main.py
import functions_framework
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModel
from pinecone import init, upsert
from google.cloud import storage
from sklearn.metrics.pairwise import cosine_similarity
import tempfile
import os

# --- Global Scope: Load the model and vectors once on cold start ---
# The model is loaded once when the function starts up, not on every call.
tokenizer = AutoTokenizer.from_pretrained('sentence-transformers/all-MiniLM-L6-v2')
model = AutoModel.from_pretrained('sentence-transformers/all-MiniLM-L6-v2')
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

# Load the vectorized data from Cloud Storage once on cold start
def download_vectors_from_gcs():
    """Downloads the numpy file from a GCS bucket."""
    client = storage.Client()
    bucket_name = 'wp-angular-ecommerce.firebasestorage.app' # Replace with your bucket name
    blob = client.bucket(bucket_name).blob('med_dialog_vectors.npy')
    
    # Use a temporary file to store the downloaded data
    temp_dir = tempfile.gettempdir()
    file_path = os.path.join(temp_dir, 'med_dialog_vectors.npy')
    
    blob.download_to_filename(file_path)
    return np.load(file_path, allow_pickle=True)

# These global variables will be available to all function invocations
med_dialog_vectors = download_vectors_from_gcs()

# You'll also need the original text to return the result
med_dialog_text = np.load('med_dialog_text.npy', allow_pickle=True)

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


@functions_framework.http
def semantic_search_function(request):
    # 1. Receive the Query
    request_json = request.get_json(silent=True)
    query_text = request_json.get('symptoms')
    if not query_text:
        return 'Symptom text is required.', 400

    # 2. Vectorize the Query
    query_vector = get_vector([query_text])

    # 3. Perform the Search (Cosine Similarity)
    # The cosine_similarity function expects a 2D array, so we wrap the query vector
    similarities = cosine_similarity(query_vector, med_dialog_vectors)

    # 4. Find the Best Match
    best_match_index = np.argmax(similarities)
    best_match_text = med_dialog_text[best_match_index]
    
    # 5. Return the Result
    return best_match_text, 200