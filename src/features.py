import pandas as pd
from tqdm import tqdm
import numpy as np
from .constants import types_dict

def analyze_defense(pokemon):
    def_types = [t for t in pokemon.get('types', []) if t != "notype"]

    multipliers = {}
    for atk_type in types_dict:
        if any(atk_type in types_dict[def_type]["immune"] for def_type in def_types):
            m = 0.0
        else:
            m = 1.0
            for def_type in def_types:
                m *= (2.0 if atk_type in types_dict[def_type]["weakness"] else 1.0)
                m *= (0.5 if atk_type in types_dict[def_type]["resistence"] else 1.0)

        multipliers[atk_type] = m
    
    buckets = {"0x":[], "1/4x":[], "1/2x":[], "2x":[], "4x":[]}
    for atk, m in multipliers.items():
        if m == 0.0: buckets["0x"].append(atk)
        elif m == 0.25: buckets["1/4x"].append(atk)
        elif m == 0.5: buckets["1/2x"].append(atk)
        elif m == 2.0: buckets["2x"].append(atk)
        elif m == 4.0: buckets["4x"].append(atk)
        else:
            pass
    return buckets

def vulnerability_score(pokemon):
    interactions = analyze_defense(pokemon)
    weights = {'0x': 4, '1/4x': 3, '1/2x': 2, '2x': -3, '4x': -6}
    return sum(len(interactions[key]) * weights[key] for key in interactions)

def static_features(battle: dict) -> dict: 

    features = {}
    stats = ["hp", "spe", "atk", "def", "spd", "spa"]

    # --- Player 1 Team Features ---
    p1_team = battle.get('p1_team_details', [])
    if p1_team:
        features['p1_def_score'] = np.mean([vulnerability_score(p) for p in p1_team])
        
        # Average stats for p1 team
        for stat in stats:
            features[f'p1_mean_{stat}'] = np.mean([p.get(f'base_{stat}', 0) for p in p1_team])

        
    # --- Player 2 Lead Features ---
    p2_lead = battle.get('p2_lead_details')
    if p2_lead:
        features['p2_def_score'] = vulnerability_score(p2_lead) 
        
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

# Get supereffective and not very effective moves 
def moves(battle):
    p1_team = battle.get('p1_team_details', [])
    battle_timeline = battle.get('battle_timeline', [])

    p1_def = {p["name"]: analyze_defense(p) for p in p1_team}

    nve_hits, se_hits = 0, 0
    prev_hp = {}

    for turn in battle_timeline:
        p1_state = turn.get('p1_pokemon_state', {})
        p2_move = turn.get('p2_move_details', {})
        if not p2_move:
            continue
        p1_name = p1_state.get('name')
        p1_hp = p1_state.get('hp_pct')
        move_type = p2_move.get('type', "")
        base_power = p2_move.get('base_power', 0)
        
        if not p1_name or not move_type or base_power <= 0:
            continue

        # Previous hp
        prev = prev_hp.get(p1_name)

        hit = (isinstance(p1_hp, (int, float)) and isinstance(prev, (int, float)) and p1_hp < prev)
        if hit:
            resistence = p1_def[p1_name]['1/2x'] + p1_def[p1_name]['1/4x']
            weakness = p1_def[p1_name]['2x'] + p1_def[p1_name]['4x']
            if move_type.lower() in weakness:
                se_hits += 1
            elif move_type.lower() in resistence:
                nve_hits += 1
        
        prev_hp[p1_name] = p1_hp
    return nve_hits, se_hits
    
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

        p1_hp = p1_pokemon_state.get('hp_pct', 1.0)
        p2_hp = p2_pokemon_state.get('hp_pct', 1.0)

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


    nve_hits, se_hits = moves(battle)
    features['p1_hp_loss'] = round(p1_hp_loss * 100, 2) 
    features['p2_hp_loss'] = round(p2_hp_loss * 100, 2)
    features['nve_hits'] = nve_hits 
    features['se_hits'] = se_hits
            
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