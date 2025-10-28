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

def display_battle(battle_id, train_data):

    # Let's inspect the first battle to see its structure
    print(f"\n--- Structure of the battle: battle_id {battle_id} ---")
    if train_data:
        first_battle = train_data[battle_id]
        
        # To keep the output clean, we can create a copy and truncate the timeline
        battle_for_display = first_battle.copy()
        battle_for_display['battle_timeline'] = battle_for_display.get('battle_timeline', [])
        
        # Use json.dumps for pretty-printing the dictionary
        print(json.dumps(battle_for_display, indent=4))
        if len(first_battle.get('battle_timeline', [])) > 3:
            print("    ...")
            print("    (battle_timeline has been truncated for display)")