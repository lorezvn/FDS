import pandas as pd
from tqdm import tqdm
import numpy as np

def types_mult(team: dict) -> dict:
    # no_effect = moves that deal 0x to my pokemons
    # def_weak = moves that deal 2x to my pokemons
    # atk_weak = moves that deal 1/2x to my pokemons
    
    types_dict = {
        "normal": {"no_effect": ["ghost"], "def_weak": ["fighting"], "atk_weak": []}, 
        "fire":   {"no_effect": [], "def_weak": ["water", "ground", "rock"], "atk_weak": ["fire", "grass", "bug"]}
    }
    pass

def static_features(battle: dict) -> dict: 

    features = {}
    stats = ["hp", "spe", "atk", "def", "spd", "spa"]

    # --- Player 1 Team Features ---
    p1_team = battle.get('p1_team_details', [])
    if p1_team:
        # Average stats for p1 team
        for stat in stats:
            features[f'p1_mean_{stat}'] = np.mean([p.get(f'base_{stat}', 0) for p in p1_team])

        

    # --- Player 2 Lead Features ---
    p2_lead = battle.get('p2_lead_details')
    if p2_lead:
        # Stats for lead pokemon p2
        for stat in stats:
            features[f'p2_lead_{stat}'] = p2_lead.get(f'base_{stat}', 0)


    # --- First turn matchup ---
    battle_timeline = battle.get('battle_timeline', [])
    if p1_team and p2_lead and battle_timeline:
        first_turn = battle_timeline[0]
        p1_pokemon_name = first_turn.get('p1_pokemon_state', {}).get('name')

        # Find matching Pokemon in p1_team
        p1_pokemon = next((p for p in p1_team if p.get('name') == p1_pokemon_name), None)

        if p1_pokemon: 
            p1_spe = p1_pokemon.get('base_spe', 0)
            p2_spe = features['p2_lead_spe']
            features['spe_diff'] = p1_spe - p2_spe
        else: 
            features['spe_diff'] = 0.0

    return features


# TODO - Risistemare
def dynamic_features(battle: dict) -> dict:

    features = {}

    p1_hp_loss = 0.0
    p2_hp_loss = 0.0
    prev_p1_hp = None
    prev_p2_hp = None
    
    battle_timeline = battle.get('battle_timeline', [])

    for turn in battle_timeline:
        p1_pokemon_state = turn.get('p1_pokemon_state', {})
        p2_pokemon_state = turn.get('p2_pokemon_state', {})
        
        p1_status = p1_pokemon_state.get('status', {})
        p2_status = p2_pokemon_state.get('status', {})

        p1_hp = p1_pokemon_state.get("hp_pct", 1.0)
        p2_hp = p2_pokemon_state.get("hp_pct", 1.0)

        # HP loss 
        if prev_p1_hp is not None:
            d = p1_hp - prev_p1_hp
            if d < 0:
                p1_hp_loss += -d
        if prev_p2_hp is not None:
            d = p2_hp - prev_p2_hp
            if d < 0:
                p2_hp_loss += -d

        prev_p1_hp = p1_hp
        prev_p2_hp = p2_hp

        # Number of fainted pokemons
        if p1_status == 'fnt': 
            key = "p1_ko_count"
            features[key] = features.get(key, 0) + 1
            
        if p2_status == 'fnt': 
            key = "p2_ko_count"
            features[key] = features.get(key, 0) + 1

        # Number of turns with altered status
        if p1_status not in ['nostatus', 'fnt']:
            key = 'p1_bad_status'
            features[key] = features.get(key, 0) + 1

        if p2_status not in ['nostatus', 'fnt']:
            key = 'p2_bad_status'
            features[key] = features.get(key, 0) + 1

    features["p1_hp_loss_total"] = round(p1_hp_loss * 100, 2)
    features["p2_hp_loss_total"] = round(p2_hp_loss * 100, 2)
            
    return features
    

def create_features(data: list[dict]) -> pd.DataFrame:
    """
    A very basic feature extraction function.
    It only uses the aggregated base stats of the player's team and opponent's lead.
    """
    feature_list = []
    for battle in tqdm(data, desc="Extracting features"):
        features = {}

        features.update(static_features(battle))
        features.update(dynamic_features(battle))

        # We also need the ID and the target variable (if it exists)
        features['battle_id'] = battle.get('battle_id')
        if 'player_won' in battle:
            features['player_won'] = int(battle['player_won'])
            
        feature_list.append(features)
        
    return pd.DataFrame(feature_list).fillna(0)