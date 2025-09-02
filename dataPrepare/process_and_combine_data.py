import os
import json
import numpy as np

# Our heuristic mapping function from our last discussion
def heuristic_map_to_standard(data_item):
    """
    Takes a single data item and maps it to a standardized format
    by guessing the field names based on keywords.
    """
    standard_item = {
        'use_case': '',
        'sample_script': ''
    }
    use_case_keywords = ['usecase', 'use_case', 'context', 'purpose', 'domain']
    script_keywords = ['script', 'sample', 'text', 'dialogue', 'response']

    for key, value in data_item.items():
        cleaned_key = key.lower().replace(' ', '_')
        if any(keyword in cleaned_key for keyword in use_case_keywords) and isinstance(value, str):
            standard_item['use_case'] = value
        
        if any(keyword in cleaned_key for keyword in script_keywords) and isinstance(value, str):
            standard_item['sample_script'] = value
            
    return standard_item

def process_and_combine_data(input_dir, output_file):
    """
    Opens all JSON files in a directory, processes them with the heuristic mapper,
    and combines the results into a single output file.
    """
    standardized_data = []
    
    # Loop through every file in the specified directory
    for filename in os.listdir(input_dir):
        if filename.endswith(".json"):
            file_path = os.path.join(input_dir, filename)
            print(f"Processing {filename}...")

            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    # Load the JSON data
                    raw_data_items = json.load(f)

                    # Check if the data is a list of dictionaries
                    if isinstance(raw_data_items, list):
                        # Map each item to our standardized format
                        for item in raw_data_items:
                            standardized_item = heuristic_map_to_standard(item)
                            # Ensure we only append valid items
                            if standardized_item['sample_script']:
                                standardized_data.append(standardized_item)
                    else:
                        print(f"Skipping {filename}: Not a list of JSON objects.")

            except json.JSONDecodeError:
                print(f"Error decoding JSON from {filename}. Skipping file.")
    
    # Save the combined, standardized data to a new JSON file
    if standardized_data:
        print(f"\nFinished processing. Saving {len(standardized_data)} items to {output_file}...")
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(standardized_data, f, indent=4, ensure_ascii=False)
        print("Success! The single output file has been created.")
    else:
        print("No valid data found. Output file not created.")

# --- How to use the script ---
if __name__ == "__main__":
    # Define your input and output paths
    input_directory = "path/to/your/json/files"
    output_filename = "standardized_data.json"
    
    process_and_combine_data(input_directory, output_filename)