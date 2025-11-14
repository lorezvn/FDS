import pandas as pd
from tqdm import tqdm
import numpy as np
from .constants import stats
from .utils import crit_rate, get_p2_team
from ..data_types import Pokedex
from .constants import types_dict

def analyze_defense(pokemon: dict) -> dict:
    """
    Compute the defensive multipliers of a Pokémon against all attacking types.
    """
    # Filter out "notype"
    def_types = [t for t in pokemon.get("types", []) if t != "notype"]

    multipliers = {}

    # Compute the multiplier for each attacking type
    for atk_type in types_dict:

        # Immunity check
        if any(atk_type in types_dict[def_t]["immune"] for def_t in def_types):
            multiplier = 0.0
        else:
            multiplier = 1.0
            for def_t in def_types:
                if atk_type in types_dict[def_t]["weakness"]:
                    multiplier *= 2.0
                if atk_type in types_dict[def_t]["resistence"]:
                    multiplier *= 0.5

        multipliers[atk_type] = multiplier

    
    mult_list = {"0x": [], "1/4x": [], "1/2x": [], "2x": [], "4x": []}
    for atk, m in multipliers.items():
        if m == 0.0: mult_list["0x"].append(atk)
        elif m == 0.25: mult_list["1/4x"].append(atk)
        elif m == 0.5: mult_list["1/2x"].append(atk)
        elif m == 2.0: mult_list["2x"].append(atk)
        elif m == 4.0: mult_list["4x"].append(atk)

    return mult_list

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
    return {
        "p1_nve_hits": nve_hits,
        "p1_se_hits": se_hits
    }

def get_type_matchup_score(p1_types: list, p2_types: list) -> float:
    """
    Computes a type advantage score for P1 over P2.

    When P1 attacks P2:
      - -3 if P2 is immune to the attack
      - +2 if P2 is weak (2x or 4x)
      - -1 if P2 resists (0.5x or 0.25x)

    When P2 attacks P1:
      - +3 if P1 is immune to the attack
      - -2 if P1 is weak (2x or 4x)
      - +1 if P1 resists (0.5x or 0.25x)
    """
    if not p1_types or not p2_types:
        return 0.0

    score = 0.0

    # P1 attacks P2
    for atk in p1_types:
        if atk not in types_dict:
            continue
        for d in p2_types:
            if d not in types_dict:
                continue

            # P2 is immune
            if atk in types_dict[d]["immune"]:
                score -= 3
            # P2 is weak
            elif atk in types_dict[d]["weakness"]:
                score += 2
            # P2 resists
            elif atk in types_dict[d]["resistence"]:
                score -= 1

    # P2 attacks P1
    for atk in p2_types:
        if atk not in types_dict:
            continue
        for d in p1_types:
            if d not in types_dict:
                continue
            
            # P1 is immune
            if atk in types_dict[d]["immune"]:
                score += 3
            # P1 is weak
            elif atk in types_dict[d]["weakness"]:
                score -= 2
            # P1 resists
            elif atk in types_dict[d]["resistence"]:
                score += 1

    return score

def speed_advantage(battle: dict, pokedex: Pokedex) -> dict:
    """
    Computes the rate at which Player 1 holds a speed advantage over Player 2 
    across all valid turns.

    The "speed advantage" is defined per turn as:
      - P1 Pokémon moves first (either due to higher move priority or higher effective speed).
      - If both Pokémon have equal Speed, the advantage is shared (0.5).

    Returned features:
      - p1_speed_adv_turns: Number of turns where P1 acted before or at the same time as P2.
      - p1_speed_adv_rate: Ratio of advantage turns over total valid turns.
    """
    
    def get_real_speed(state, pokedex):
        name = state.get('name')
        if not name or name not in pokedex:
            return 0.0

        base_speed = pokedex[name].get('base_spe', 0)
        boosts = state.get('boosts', {})
        stage = boosts.get('spe', 0)
        status = state.get('status')

        # Speed mult
        mult = (2 + stage) / 2 if stage >= 0 else 2 / (2 - stage)

        effective_speed = base_speed * mult

        # Paralysis penality (Gen 1: x0.25)
        if status == 'par': effective_speed *= 0.25

        return effective_speed

    battle_timeline = battle.get('battle_timeline', [])
    adv, v_turns = 0, 0

    for turn in battle_timeline:

        p1_state = turn.get('p1_pokemon_state', {})
        p2_state = turn.get('p2_pokemon_state', {})
        p1_move = turn.get('p1_move_details', {})
        p2_move = turn.get('p2_move_details', {})

        # No attacks (no valid turn for computing speed adv)
        if not p1_move and not p2_move:
            continue

        v_turns += 1

        # Just one pokemon attacks
        if p1_move and not p2_move:
            adv += 1
            continue
        elif not p1_move and p2_move:
            continue 

        # Two pokemon attacks
        p1_priority = p1_move.get('priority', 0)
        p2_priority = p2_move.get('priority', 0)

        if p1_priority > p2_priority:
            adv += 1
        elif p1_priority < p2_priority:
            continue
        else:
            p1_speed = get_real_speed(p1_state, pokedex)
            p2_speed = get_real_speed(p2_state, pokedex)

            if p1_speed > p2_speed:
                adv += 1
            elif p1_speed == p2_speed:
                adv += 0.5
            else: continue
    
    
    # Speed advantage rate
    rate = round(adv / v_turns, 3) if v_turns > 0 else 0.0

    features = {
        #'p1_speed_adv_turns': adv, 
        'p1_speed_adv_rate': rate
    }

    return features


def switch_dynamics_features(battle: dict) -> dict:
    """
    Analyze switch (substitution) behavior for P1 and P2.
    
    A switch can be categorized as:
      - voluntary: the player switched without using a move
      - forced_faint: the player's active Pokémon fainted (status 'fnt')
    
    Ritorna un dizionario con i conteggi totali per ciascuna categoria.
    """
    

    # Timeline dei turni della battaglia
    tl = battle.get('battle_timeline', []) or []

    # Se la battaglia ha meno di 2 turni, non può esserci stato alcun cambio
    if len(tl) < 2:
        return {
            'p1_switch_count': 0, 'p2_switch_count': 0,
            'p1_voluntary_switches': 0, 'p2_voluntary_switches': 0,
            'p1_forced_faint_switches': 0, 'p2_forced_faint_switches': 0,
            'p1_switch_rate': 0.0, 'p2_switch_rate': 0.0,
        }

    # --- Inizializza stato del turno precedente ---
    # Nome e stato (status) dei Pokémon iniziali
    p1_prev_name = tl[0].get('p1_pokemon_state', {}).get('name')
    p2_prev_name = tl[0].get('p2_pokemon_state', {}).get('name')
    p1_prev_status = (tl[0].get('p1_pokemon_state', {}) or {}).get('status') or 'nostatus'
    p2_prev_status = (tl[0].get('p2_pokemon_state', {}) or {}).get('status') or 'nostatus'

    # --- Contatori per ogni tipo di cambio ---
    p1_sw = p2_sw = 0   # cambi totali
    p1_vol = p2_vol = 0 # cambi volontari
    p1_fnt = p2_fnt = 0 # cambi forzati da KO
    total_turns = len(tl)

    # --- Loop sui turni successivi ---
    for t in range(1, total_turns):
        # Stato corrente del turno t
        turn = tl[t]
        p1s = turn.get('p1_pokemon_state', {}) or {}
        p2s = turn.get('p2_pokemon_state', {}) or {}

        # Pokémon attuali e stati
        p1_name = p1s.get('name')
        p2_name = p2s.get('name')
        p1_status = p1s.get('status') or 'nostatus'
        p2_status = p2s.get('status') or 'nostatus'

        # Mossa usata in questo turno (può essere None se ha solo cambiato)
        p1_move = turn.get('p1_move_details')
        p2_move = turn.get('p2_move_details')

        # Verifica se è avvenuto un cambio (nome Pokémon diverso da quello precedente)
        p1_switch = (p1_prev_name is not None and p1_name and p1_name != p1_prev_name)
        p2_switch = (p2_prev_name is not None and p2_name and p2_name != p2_prev_name)

        # --- Classificazione cambio P1 ---
        if p1_switch:
            p1_sw += 1  # incrementa i cambi totali
            if p1_prev_status == 'fnt':
                # il Pokémon precedente è morto → cambio forzato
                p1_fnt += 1
            elif p1_move is None:
                # non ha usato mosse → cambio volontario
                p1_vol += 1
            

        # --- Classificazione cambio P2 ---
        if p2_switch:
            p2_sw += 1
            if p2_prev_status == 'fnt':
                p2_fnt += 1
            elif p2_move is None:
                p2_vol += 1
            

        # Aggiorna "stato precedente" per il prossimo turno
        p1_prev_name = p1_name or p1_prev_name
        p2_prev_name = p2_name or p2_prev_name
        p1_prev_status = p1_status
        p2_prev_status = p2_status

    # --- Output finale ---
    # Restituisce tutti i conteggi raccolti
    return {
        'p1_switch_count': p1_sw,
        'p2_switch_count': p2_sw,
        'p1_voluntary_switches': p1_vol,
        'p2_voluntary_switches': p2_vol,
        'p1_forced_faint_switches': p1_fnt,
        'p2_forced_faint_switches': p2_fnt,
    }


def status_advantages(battle: dict) -> dict:
    """
    Extracts status-related ('frz' and 'par') features.

    Returned features:
      - p1_frz_turns, p2_frz_turns:
          Number of turns each team spent frozen ('frz').

      - frz_turns_diff:
          p2_frz - p1_frz (positive if P2 was frozen longer: advantage for P1).

      - early_paralysis_lead:
          (1 / first_par_turn_P1) - (1 / first_par_turn_P2)
          Higher values indicate P1 inflicted paralysis earlier.
    """
    battle_timeline = battle.get('battle_timeline', [])
    
    def analyze_status(player: str):

        frz_turns = 0
        first_par_turn = None

        for idx, turn in enumerate(battle_timeline):

            t = turn.get('turn') or (idx + 1)

            status = turn.get(f'{player}_pokemon_state', {}).get('status')

            # Frozen turn
            if status == 'frz': frz_turns += 1

            # First time paralized
            if status == 'par' and first_par_turn is None:
                first_par_turn = int(t)

        first_par_inv = (1.0 / first_par_turn) if (first_par_turn and first_par_turn > 0) else 0.0
        return frz_turns, first_par_inv

    p1_frz, p1_par_inv = analyze_status('p1')
    p2_frz, p2_par_inv = analyze_status('p2')

    features = {
        'p1_frz_turns': p1_frz,
        'p2_frz_turns': p2_frz,
        'frz_turns_diff': p2_frz - p1_frz,
        'early_paralysis_lead': p1_par_inv - p2_par_inv,
    }

    return features


def offensive_rate(battle: dict) -> dict:
    """
    Computes offensive tendency metrics for both players based on the frequency 
    of attacking moves used during the battle.

    An "offensive move" is defined as any move with 'base_power' > 0.

    Returned features:
      - p1_offensive_rate: Fraction of turns where P1 used an offensive move.
      - p2_offensive_rate: Fraction of turns where P2 used an offensive move.
      - offensive_diff: Difference (P1 - P2)
    """

    battle_timeline = battle.get('battle_timeline', [])

    p1_off_moves = 0  
    p2_off_moves = 0  
    total_turns = 0   

    for turn in battle_timeline:
        total_turns += 1 

        p1_move = turn.get('p1_move_details')
        p2_move = turn.get('p2_move_details')

        # Offensive move -> 'base_power' > 0
        if p1_move and p1_move.get('base_power', 0) > 0:
            p1_off_moves += 1
        if p2_move and p2_move.get('base_power', 0) > 0:
            p2_off_moves += 1

    p1_rate = p1_off_moves / total_turns
    p2_rate = p2_off_moves / total_turns

    features = {
        'p1_offensive_rate': p1_rate, 
        'p2_offensive_rate': p2_rate, 
        'offensive_diff': p1_rate - p2_rate  
    }

    return features


def extract_status_features(battle: dict) -> dict:
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


def static_features(battle: dict, pokedex: Pokedex) -> dict: 

    features = {}

    # --- Player 1 Team Features ---
    p1_team = battle.get('p1_team_details', [])
    if p1_team:
        features['p1_avg_crit_rate'] = np.mean([crit_rate(p.get('base_spe', 0)) for p in p1_team])
        
        # Average stats for p1 team
        for stat in stats:
            features[f'p1_mean_{stat}'] = np.mean([p.get(f'base_{stat}', 0) for p in p1_team])

        
    # --- Player 2 Observed Team Features ---
    p2_lead = battle.get('p2_lead_details')
    p2_team = get_p2_team(battle, pokedex)
    
    if p2_team:
        features['p2_avg_crit_rate'] = np.mean([crit_rate(p.get('base_spe', 0)) for p in p2_team])
        
        # Average stats for observed p2_team
        for stat in stats:
            features[f'p2_mean_{stat}'] = np.mean([p.get(f'base_{stat}', 0) for p in p2_team])

        # Team coverage
        features["p2_team_coverage"] = min(len(p2_team) / 6.0, 1.0)


    # --- Average stats differences
    for stat in stats:
        p1_mean = features.get(f'p1_mean_{stat}', 0)
        p2_mean = features.get(f'p2_mean_{stat}', 0)
        features[f'mean_{stat}_diff'] = p1_mean - p2_mean


    # --- First turn matchup ---
    battle_timeline = battle.get('battle_timeline', [])
    if p1_team and p2_lead and battle_timeline:
        first_turn = battle_timeline[0]
        p1_pokemon_name = first_turn.get('p1_pokemon_state', {}).get('name')

        # Find matching Pokemon in p1_team
        p1_lead = next((p for p in p1_team if p.get('name') == p1_pokemon_name), None)

        if p1_lead: 
            p1_spe = p1_lead.get('base_spe', 0)
            p2_spe = p2_lead.get('base_spe', 0)
            features['spe_diff'] = p1_spe - p2_spe
        else: 
            features['spe_diff'] = 0.0

    # --- Matchup type advantage ---
    if p1_team and p2_team:
        team_adv_score = 0
        total_matchups = 0
        for p1_pokemon in p1_team:
            for p2_pokemon in p2_team:
                p1_types = p1_pokemon.get('types', [])
                p2_types = p2_pokemon.get('types', [])
                team_adv_score += get_type_matchup_score(p1_types, p2_types)
                total_matchups += 1
        
        features['team_type_adv_mean'] = round(team_adv_score / total_matchups, 3) if total_matchups > 0 else 0.0
        
    return features
    
def dynamic_features(battle: dict) -> dict:

    features = {
        'p1_bad_status': 0, 'p2_bad_status': 0,
        'p1_ko_count': 0, 'p2_ko_count': 0
    }

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
            features['p1_bad_status'] += 1

        if p2_status not in ['nostatus', 'fnt']:
            features['p2_bad_status'] += 1

        # Number of fainted pokemons
        if p1_status == 'fnt': 
            features['p1_ko_count'] += 1
            
        if p2_status == 'fnt': 
            features['p2_ko_count'] += 1

    return features
    

def create_features(data: list[dict], pokedex: Pokedex) -> pd.DataFrame:
    feature_list = []
    for battle in tqdm(data, desc="Extracting features"):
        #if battle.get('battle_id') == 4877: continue
        
        features = {}

        features.update(speed_advantage(battle, pokedex))
        features.update(switch_dynamics_features(battle))
        features.update(status_advantages(battle)) 
        features.update(offensive_rate(battle))
        features.update(extract_status_features(battle)) 
        features.update(static_features(battle, pokedex))
        features.update(dynamic_features(battle))
        features.update(moves(battle))

        # We also need the ID and the target variable (if it exists)
        features['battle_id'] = battle.get('battle_id')
        if 'player_won' in battle:
            features['player_won'] = int(battle['player_won'])
            
        feature_list.append(features)
        
    return pd.DataFrame(feature_list).fillna(0)