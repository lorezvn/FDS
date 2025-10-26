import json
import os

def load_jsonl(data_path: str) -> list:
    if not os.path.exists(data_path):
        return FileNotFoundError(f"ERROR: Could not find the file at '{data_path}'.")
    
    data = []
    with open(data_path, 'r') as f:
        for line in f:
            # json.loads() parses one line (one JSON object) into a Python dictionary
            data.append(json.loads(line))
            
    if not data:
        raise ValueError(f"ERROR: File {data_path} invalid.")
    
    return data