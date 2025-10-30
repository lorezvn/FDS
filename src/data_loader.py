import json
import os
from tqdm import tqdm
from .features.constants import stats
from .data_types import Pokedex

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


def display_battle(battle_id: int, train_data: list[dict]):

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


def create_pokedex(data: list[dict]) -> Pokedex:
    pokedex = {}
    for battle in tqdm(data, desc="Creating pokedex"):
        p1_team = battle.get('p1_team_details', [])
        p2_lead = battle.get('p2_lead_details')
        pokemon_list = p1_team + ([p2_lead] if p2_lead else [])

        for pokemon in pokemon_list:
            if not pokemon:
                continue
            pokemon_name = pokemon.get('name')
            if pokemon_name and pokemon_name not in pokedex:
                pokemon_stats = {f'base_{stat}': pokemon.get(f'base_{stat}') for stat in stats}
                pokemon_types = [t.lower() for t in (pokemon.get('types') or [])]
                pokemon_stats['types'] = pokemon_types
                pokedex[pokemon_name] = pokemon_stats

    return pokedex