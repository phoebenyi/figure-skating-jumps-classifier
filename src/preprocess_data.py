import json
import numpy as np

import json
import numpy as np
import os

def load_and_preprocess_json(dir_path):
    keypoints_list = []
    
    for skater in os.listdir(dir_path):
        skater_path = os.path.join(dir_path, skater)
        
        if os.path.isdir(skater_path):
            for jump_type in os.listdir(skater_path):
                jump_path = os.path.join(skater_path, jump_type)
                
                if os.path.isdir(jump_path):
                    for json_file in os.listdir(jump_path):
                        if json_file.endswith(".json"):
                            file_path = os.path.join(jump_path, json_file)
                            
                            with open(file_path, 'r') as f:
                                try:
                                    data = json.load(f)
                                    keypoints = np.array(data['keypoints'])
                                    keypoints_list.append(keypoints)
                                except Exception as e:
                                    print(f"Error reading {file_path}: {e}")
    
    keypoints_array = np.array(keypoints_list)
    keypoints_normalized = (keypoints_array - np.mean(keypoints_array, axis=0)) / np.std(keypoints_array, axis=0)
    
    return keypoints_normalized

if __name__ == "__main__":
    path = "../data/json"
    preprocessed_data = load_and_preprocess_json(path)
    np.save("../data/preprocessed_data.npy", preprocessed_data)
