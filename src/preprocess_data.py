import os
import json
import numpy as np
import pandas as pd
from tensorflow.keras.preprocessing.sequence import pad_sequences

def preprocess_data(json_directory, output_file):
    data = []
    labels = []
    
    for skater_folder in os.listdir(json_directory):
        skater_path = os.path.join(json_directory, skater_folder)

        # Skip if it's not a directory
        if not os.path.isdir(skater_path):
            print(f'Skipping non-directory: {skater_path}')
            continue

        for jump_type in os.listdir(skater_path):
            jump_path = os.path.join(skater_path, jump_type)

            # Skip if it's not a directory
            if not os.path.isdir(jump_path):
                print(f'Skipping non-directory: {jump_path}')
                continue

            for file in os.listdir(jump_path):
                if file.endswith('.json'):
                    file_path = os.path.join(jump_path, file)
                    print(f'Processing: {file_path}')
                    try:
                        with open(file_path, 'r', encoding='ISO-8859-1') as f:
                            json_data = json.load(f)
                            markers = json_data.get('Markers', [])

                            if not markers:
                                print(f'No markers found in {file_path}')
                                continue

                            # Extract relevant marker data for features
                            for marker in markers:
                                values = marker['Parts'][0]['Values']
                                # Flatten values and append to the data list
                                data.append(np.array(values).flatten())
                                labels.append(jump_type)

                    except Exception as e:
                        print(f'Error processing {file_path}: {e}')

    # Pad sequences to ensure uniform input size
    max_length = max(len(seq) for seq in data)  # Get max length dynamically
    X = pad_sequences(data, maxlen=max_length, padding='post')  # Pad sequences
    y = pd.get_dummies(labels).values  # One-hot encode labels

    # Save preprocessed data
    np.savez(output_file, X=X, y=y)
    print(f'Saved processed data to {output_file}')

# Run the preprocessing
preprocess_data('../data/json/', 'processed_data.npz')
