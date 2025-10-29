import pandas as pd
from tqdm import tqdm
import numpy as np
from .constants import stats

def crit_rate(base_speed):
    rate = base_speed * 100 / 512
    return round(rate, 4)

def static_features(battle: dict) -> dict: 

    features = {}

    # --- Player 1 Team Features ---
    p1_team = battle.get('p1_team_details', [])
    features['p1_team_diversity'] = len(set(t for p in p1_team for t in p.get('types', []) if t != "notype"))
    if p1_team:
        features['p1_avg_crit_rate'] = np.mean([crit_rate(p.get('base_spe', 0)) for p in p1_team])
        
        # Average stats for p1 team
        for stat in stats:
            features[f'p1_mean_{stat}'] = np.mean([p.get(f'base_{stat}', 0) for p in p1_team])

        
    # --- Player 2 Lead Features ---
    p2_lead = battle.get('p2_lead_details')
    if p2_lead:
        features['p2_crit_rate'] = crit_rate(p2_lead.get('base_spe', 0))
        
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

# status info with weights 
def extract_status_features(battle):
    status_weights = {"slp": 3, "frz": 4, "par": 2,"tox": 1.5, "psn": 1,"brn": 0.5}

    features = {}

    battle_timeline = battle.get('battle_timeline', [])
    p1_score = 0.0
    p2_score = 0.0

    for turn in battle_timeline:
        p1_status = turn.get('p1_pokemon_state', {}).get('status')
        p2_status = turn.get('p2_pokemon_state', {}).get('status')

        if p1_status in status_weights:
            p1_score += status_weights[p1_status]
        if p2_status in status_weights:
            p2_score += status_weights[p2_status]

    features['status_diff'] = p1_score - p2_score
    return features
    
# TODO - Risistemare
def dynamic_features(battle: dict) -> dict:

    features = {'p1_ko_count': 0, 'p2_ko_count': 0}

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

        features['p1_hp_loss'] = round(p1_hp_loss * 100, 2) 
        features['p2_hp_loss'] = round(p2_hp_loss * 100, 2)

        # Number of turns with altered status
        if p1_status not in ['nostatus', 'fnt']:
            key = 'p1_bad_status'
            features[key] = features.get(key, 0) + 1

        if p2_status not in ['nostatus', 'fnt']:
            key = 'p2_bad_status'
            features[key] = features.get(key, 0) + 1

        # Number of fainted pokemons
        if p1_status == 'fnt': 
            features['p1_ko_count'] += 1
            
        if p2_status == 'fnt': 
            features['p2_ko_count'] += 1
    

def create_features(data: list[dict]) -> pd.DataFrame:
    """
    A very basic feature extraction function.
    It only uses the aggregated base stats of the player's team and opponent's lead.
    """
    feature_list = []
    for battle in tqdm(data, desc="Extracting features"):
        if battle.get('battle_id') == 4877: continue
        
        features = {}

        features.update(extract_status_features(battle))
        features.update(static_features(battle))
        features.update(dynamic_features(battle))

        # We also need the ID and the target variable (if it exists)
        features['battle_id'] = battle.get('battle_id')
        if 'player_won' in battle:
            features['player_won'] = int(battle['player_won'])
            
        feature_list.append(features)
        
    return pd.DataFrame(feature_list).fillna(0)