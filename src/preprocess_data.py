import json
import numpy as np

def load_and_preprocess_json(path):
    with open(path, 'r', encoding='ISO-8859-1') as f:
        data = json.load(f)
    # Process the data to extract 3D pose keypoints
    # Assume data['keypoints'] is the part where 3D keypoints are stored
    keypoints = np.array([entry['keypoints'] for entry in data])
    # Normalize the keypoints if needed
    keypoints_normalized = (keypoints - np.mean(keypoints, axis=0)) / np.std(keypoints, axis=0)
    return keypoints_normalized

if __name__ == "__main__":
    path = "../data/json/alldata.json"
    preprocessed_data = load_and_preprocess_json(path)
    np.save("data/preprocessed_data.npy", preprocessed_data)

