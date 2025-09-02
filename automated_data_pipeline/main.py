# main.py
import json
import functions_framework
import numpy as np
import torch
import os
import tempfile
from transformers import AutoTokenizer, AutoModel
from pinecone import init, Index, upsert
from google.cloud import storage

# --- Global Scope: Load resources once ---
tokenizer = AutoTokenizer.from_pretrained('sentence-transformers/all-MiniLM-L6-v2')
model = AutoModel.from_pretrained('sentence-transformers/all-MiniLM-L6-v2')
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

# Initialize Pinecone client
init(api_key=os.getenv('PINECONE_API_KEY'), environment=os.getenv('PINECONE_ENVIRONMENT'))
index_name = 'aihuman-core' # Name of your Pinecone index
# Get the index instance once in the global scope
index = Index(index_name)

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

def heuristic_map_to_standard(data_item):
    """
    Takes a single data item and maps it to a standardized format
    by guessing the field names based on keywords.
    """
    standard_item = {
        'use_case': '',
        'sample_script': ''
    }

    # Define a list of keywords to look for
    use_case_keywords = ['usecase', 'use_case', 'context', 'purpose', 'domain']
    script_keywords = ['script', 'sample', 'text', 'dialogue', 'response']
    
    # Iterate through all keys in the data item
    for key, value in data_item.items():
        # Clean the key for comparison (e.g., lowercase and remove spaces)
        cleaned_key = key.lower().replace(' ', '_')

        # Check if the key matches our keywords
        if any(keyword in cleaned_key for keyword in use_case_keywords):
            standard_item['use_case'] = value
        
        if any(keyword in cleaned_key for keyword in script_keywords):
            standard_item['sample_script'] = value

    return standard_item

@functions_framework.cloud_event
def process_new_data(cloud_event):
    """
    Triggered by a new file in a Cloud Storage bucket.
    """
    data = cloud_event.data
    bucket_name = data['bucket']
    file_name = data['name']
    
    # 1. Download the new file from the bucket
    storage_client = storage.Client()
    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(file_name)
    
    # Use a temporary file to store the downloaded data
    temp_dir = os.path.join(tempfile.gettempdir(), 'new_data')
    os.makedirs(temp_dir, exist_ok=True)
    file_path = os.path.join(temp_dir, file_name)
    blob.download_to_filename(file_path)

    texts_to_vectorize = []

    # Check the file type and process accordingly
    if file_name.endswith('.json'):
        with open(file_path, 'r') as f:
            raw_data_items = json.load(f)
            standardized_data = [heuristic_map_to_standard(item) for item in raw_data_items]
            texts_to_vectorize = [f"{item['use_case']} {item['sample_script']}" for item in standardized_data]
            
    elif file_name.endswith('.txt'):
        with open(file_path, 'r') as f:
            # For a simple text file, each line could be a separate entry
            texts_to_vectorize = [line.strip() for line in f.readlines()]
            
    else:
        print(f"File type not supported: {file_name}. Skipping.")
        return 'Unsupported file type', 400

    # Ensure we have data to vectorize
    if not texts_to_vectorize:
        print("No valid data found to vectorize.")
        return 'No valid data', 400

    # --- Vectorize and Upsert to Pinecone ---
    new_vectors = get_vector(texts_to_vectorize)
    vectors_to_upsert = []
    for i, vector in enumerate(new_vectors):
        vectors_to_upsert.append( (f'{file_name}-{i}', vector.tolist(), {"original_text": texts_to_vectorize[i]}) )
    
    index.upsert(vectors=vectors_to_upsert)

    print(f"Successfully processed {file_name} and upserted {len(vectors_to_upsert)} vectors to Pinecone.")
    return 'OK', 200

    # 2. Process the file (e.g., read a JSON file into text)
    # For this blueprint, we'll assume the file is a text file with one item per line
    with open(temp_file_path, 'r') as f:
        new_data_texts = f.readlines()

    # 3. Vectorize the new data
    new_vectors = get_vector(new_data_texts)

    # 4. Upsert the new vectors into the Pinecone index
    # We'll need to create unique IDs for each vector
    index = init(index_name)
    vectors_to_upsert = []
    for i, vector in enumerate(new_vectors):
        vectors_to_upsert.append( (f'{file_name}-{i}', vector.tolist()) ) # Pinecone expects a list
    
    index.upsert(vectors=vectors_to_upsert)

    print(f"Successfully processed {file_name} and upserted {len(vectors_to_upsert)} vectors to Pinecone.")
    return 'OK', 200