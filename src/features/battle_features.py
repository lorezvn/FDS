import pandas as pd
from tqdm import tqdm
import numpy as np
from .constants import stats
from .utils import crit_rate, get_p2_team, get_type_matchup_score, analyze_defense
from ..data_types import Pokedex

def moves(battle: dict) -> dict:
    """
    Counts how many times Player 1's Pokémon were hit by 
    super-effective or not-very-effective moves.

    For each turn, if Player 2's move causes Player 1's active Pokémon
    to lose HP, the move is categorized as:
      - Super effective (SE)   → move type is in 2x or 4x weaknesses.
      - Not very effective (NVE) → move type is in 1/2x or 1/4x resistances.

    Returned features:
      - p1_nve_hits: Number of hits where P1 was struck by a resisted move.
      - p1_se_hits:  Number of hits where P1 was struck by a super-effective move.
    """

    p1_team = battle.get('p1_team_details', [])
    battle_timeline = battle.get('battle_timeline', [])

    # Precompute defensive profiles for each P1 team member
    p1_def = {p["name"]: analyze_defense(p) for p in p1_team if "name" in p}

    nve_hits, se_hits = 0, 0
    prev_hp = {}  # Track previous HP% for each Pokémon to detect actual damage

    for turn in battle_timeline:
        p1_state = turn.get('p1_pokemon_state', {})
        p2_move = turn.get('p2_move_details', {})

        # If P2 did not use a move this turn, nothing to log
        if not p2_move:
            continue

        p1_name = p1_state.get('name')
        p1_hp = p1_state.get('hp_pct')
        move_type = p2_move.get('type', "")
        base_power = p2_move.get('base_power', 0)

        # Skip if we lack essential information or move is non-damaging
        if not p1_name or not move_type or base_power <= 0:
            continue

        # Previous HP to check if damage was actually taken this turn
        prev = prev_hp.get(p1_name)

        hit = (
            isinstance(p1_hp, (int, float)) and
            isinstance(prev, (int, float)) and
            p1_hp < prev
        )

        if hit and p1_name in p1_def:
            resistence = p1_def[p1_name]['1/2x'] + p1_def[p1_name]['1/4x']
            weakness = p1_def[p1_name]['2x'] + p1_def[p1_name]['4x']

            # Normalize type name for matching
            move_t = move_type.lower()

            if move_t in [t.lower() for t in weakness]:
                se_hits += 1
            elif move_t in [t.lower() for t in resistence]:
                nve_hits += 1

        # Update stored HP for the next turn
        prev_hp[p1_name] = p1_hp

    return {
        "p1_nve_hits": nve_hits,
        "p1_se_hits": se_hits
    }

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
    Analyzes switching (substitution) behavior for Player 1 (P1) and Player 2 (P2).

    A switch is detected whenever the active Pokémon's name changes between
    consecutive turns. Each switch is categorized as:
      - voluntary: the player switched without selecting a move that turn
                   (move_details is None)
      - forced_faint: the previous active Pokémon had status 'fnt' (fainted)
                      at the time of the switch

    Returned features:
      - p1_switch_count, p2_switch_count:
            Total number of switches performed by P1 and P2.
      - p1_voluntary_switches, p2_voluntary_switches:
            Number of voluntary switches.
      - p1_forced_faint_switches, p2_forced_faint_switches:
            Number of switches forced by a fainted Pokémon.
      - p1_switch_rate, p2_switch_rate:
            Switches per turn over the whole battle timeline.
    """

    # Battle timeline (list of turns)
    battle_timeline = battle.get('battle_timeline', []) or []

    # --- Initialize "previous turn" state ---
    # Initial active Pokémon names and statuses
    p1_prev_state = battle_timeline[0].get('p1_pokemon_state', {}) or {}
    p2_prev_state = battle_timeline[0].get('p2_pokemon_state', {}) or {}

    p1_prev_name = p1_prev_state.get('name')
    p2_prev_name = p2_prev_state.get('name')
    p1_prev_status = p1_prev_state.get('status') or 'nostatus'
    p2_prev_status = p2_prev_state.get('status') or 'nostatus'


    p1_sw = p2_sw = 0   # total switches
    p1_vol = p2_vol = 0 # voluntary switches
    p1_fnt = p2_fnt = 0 # switches forced by faint
    total_turns = len(battle_timeline)

    # --- Iterate over subsequent turns ---
    for t in range(1, total_turns):
        turn = battle_timeline[t]

        # Current states for this turn
        p1s = turn.get('p1_pokemon_state', {}) or {}
        p2s = turn.get('p2_pokemon_state', {}) or {}

        p1_name = p1s.get('name')
        p2_name = p2s.get('name')
        p1_status = p1s.get('status') or 'nostatus'
        p2_status = p2s.get('status') or 'nostatus'

        # Moves used this turn (can be None if the player only switched)
        p1_move = turn.get('p1_move_details')
        p2_move = turn.get('p2_move_details')

        # Check if a switch occurred (current Pokémon name differs from previous one)
        p1_switch = (p1_prev_name is not None and p1_name and p1_name != p1_prev_name)
        p2_switch = (p2_prev_name is not None and p2_name and p2_name != p2_prev_name)

        # --- Classify P1 switch ---
        if p1_switch:
            p1_sw += 1  # total switches
            if p1_prev_status == 'fnt':
                # Previous Pokémon had fainted -> forced switch
                p1_fnt += 1
            elif p1_move is None:
                # No move used this turn -> voluntary switch
                p1_vol += 1

        # --- Classify P2 switch ---
        if p2_switch:
            p2_sw += 1
            if p2_prev_status == 'fnt':
                p2_fnt += 1
            elif p2_move is None:
                p2_vol += 1

        # Update "previous" state for the next iteration
        p1_prev_name = p1_name or p1_prev_name
        p2_prev_name = p2_name or p2_prev_name
        p1_prev_status = p1_status
        p2_prev_status = p2_status

    # Final output with all collected counts and rates
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
    """
    Computes a status-based score for each player and returns the difference 
    (P1_score - P2_score).

    Each non-volatile status condition contributes a weighted penalty:
        - slp (sleep):        3
        - frz (freeze):       4
        - par (paralysis):    2
        - tox (bad poison):   1.5
        - psn (poison):       1
        - brn (burn):         0.5

    For every turn, the active Pokémon's status contributes to its player's 
    cumulative score. A higher score means the player suffered more from 
    status conditions across the battle.

    Returned features:
        - status_diff: Total P1_status_score - P2_status_score
    """

    status_weights = {
        "slp": 3, 
        "frz": 4, 
        "par": 2,
        "tox": 1.5, 
        "psn": 1,
        "brn": 0.5
    }

    battle_timeline = battle.get('battle_timeline', []) or []

    p1_score = 0.0
    p2_score = 0.0

    for turn in battle_timeline:
        # Status of each player's active Pokémon
        p1_status = turn.get('p1_pokemon_state', {}).get('status')
        p2_status = turn.get('p2_pokemon_state', {}).get('status')

        # Add corresponding weights (if the status is a tracked non-volatile condition)
        if p1_status in status_weights:
            p1_score += status_weights[p1_status]
        if p2_status in status_weights:
            p2_score += status_weights[p2_status]

    return {
        "status_diff": p1_score - p2_score
    }


def static_features(battle: dict, pokedex: Pokedex) -> dict:
    """
    Computes static, team-level features for both Player 1 (P1) and Player 2 (P2).
    These features do not depend on turn-by-turn battle events, but instead on the 
    team compositions as inferred from the battle data.

    Extracted information includes:
        - Average critical-hit rate per team
        - Average base stats (HP, Atk, Def, Spe, etc.) for P1 and P2
        - Differences between the teams' average stats
        - First-turn Speed comparison between the two leading Pokémon
        - Average type-advantage score between every P1-P2 Pokémon pair
        - Estimated P2 team coverage (how much of its team has been revealed)

    Returned features include:
        - p1_avg_crit_rate, p2_avg_crit_rate
        - p1_mean_<stat>, p2_mean_<stat>
        - mean_<stat>_diff
        - spe_diff (lead speed difference)
        - team_type_adv_mean
        - p2_team_coverage
    """

    features = {}

    # --- Player 1 Team Features ---
    p1_team = battle.get('p1_team_details', [])
    if p1_team:
        # Average crit rate for P1 team
        features['p1_avg_crit_rate'] = np.mean([
            crit_rate(p.get('base_spe', 0)) for p in p1_team
        ])

        # Average base stats for each stat category
        for stat in stats:
            features[f'p1_mean_{stat}'] = np.mean([
                p.get(f'base_{stat}', 0) for p in p1_team
            ])

    # --- Player 2 Observed Team Features ---
    p2_lead = battle.get('p2_lead_details')
    p2_team = get_p2_team(battle, pokedex)

    if p2_team:
        # Average crit rate for observed P2 team
        features['p2_avg_crit_rate'] = np.mean([
            crit_rate(p.get('base_spe', 0)) for p in p2_team
        ])

        # Average stats for observed P2 team
        for stat in stats:
            features[f'p2_mean_{stat}'] = np.mean([
                p.get(f'base_{stat}', 0) for p in p2_team
            ])

        # Fraction of P2's team observed (capped at 1.0)
        features["p2_team_coverage"] = min(len(p2_team) / 6.0, 1.0)

    # --- Stat Differences (P1 − P2) ---
    for stat in stats:
        p1_mean = features.get(f'p1_mean_{stat}', 0)
        p2_mean = features.get(f'p2_mean_{stat}', 0)
        features[f'mean_{stat}_diff'] = p1_mean - p2_mean

    # --- First Turn Matchup Speed Difference ---
    battle_timeline = battle.get('battle_timeline', [])
    if p1_team and p2_lead and battle_timeline:
        first_turn = battle_timeline[0]
        p1_active_name = first_turn.get('p1_pokemon_state', {}).get('name')

        # Identify which Pokémon P1 started with
        p1_lead = next((p for p in p1_team if p.get('name') == p1_active_name), None)

        if p1_lead:
            p1_spe = p1_lead.get('base_spe', 0)
            p2_spe = p2_lead.get('base_spe', 0)
            features['spe_diff'] = p1_spe - p2_spe
        else:
            features['spe_diff'] = 0.0

    # --- Team vs. Team Type Advantage ---
    if p1_team and p2_team:
        team_adv_score = 0
        total_matchups = 0

        # Compute type advantage for each P1–P2 Pokémon pairing
        for p1_pokemon in p1_team:
            for p2_pokemon in p2_team:
                p1_types = p1_pokemon.get('types', [])
                p2_types = p2_pokemon.get('types', [])
                team_adv_score += get_type_matchup_score(p1_types, p2_types)
                total_matchups += 1

        features['team_type_adv_mean'] = (
            round(team_adv_score / total_matchups, 3)
            if total_matchups > 0 else 0.0
        )

    return features

    
def dynamic_features(battle: dict) -> dict:
    """
    Computes dynamic, time-dependent features from the battle timeline.

    For each turn, the function tracks:
      - HP loss accumulated over the whole battle for P1 and P2
      - Number of turns in which each player had a non-neutral status
      - Number of times a Pokémon fainted for each player

    Assumptions:
      - 'hp_pct' is the current HP fraction in [0, 1] for the active Pokémon.
      - 'status' can be:
            'nostatus' for healthy,
            'fnt' for fainted,
            or any other non-volatile status (paralysis, burn, etc.).

    Returned features:
      - p1_hp_loss, p2_hp_loss: total HP percentage points lost (0-100 scale)
      - p1_bad_status, p2_bad_status: number of turns with a non-neutral status
      - p1_ko_count, p2_ko_count: number of faint events observed
    """

    features = {
        'p1_bad_status': 0,
        'p2_bad_status': 0,
        'p1_ko_count': 0,
        'p2_ko_count': 0
    }

    p1_hp_loss = 0.0
    p2_hp_loss = 0.0
    prev_p1_hp = None
    prev_p2_hp = None

    battle_timeline = battle.get('battle_timeline', []) or []

    for turn in battle_timeline:
        p1_pokemon_state = turn.get('p1_pokemon_state', {}) or {}
        p2_pokemon_state = turn.get('p2_pokemon_state', {}) or {}

        # Current status (default to 'nostatus' if missing)
        p1_status = p1_pokemon_state.get('status') or 'nostatus'
        p2_status = p2_pokemon_state.get('status') or 'nostatus'

        # Current HP percentage (default to full HP)
        p1_hp = p1_pokemon_state.get('hp_pct', 1.0)
        p2_hp = p2_pokemon_state.get('hp_pct', 1.0)

        # --- HP loss tracking ---
        if isinstance(p1_hp, (int, float)) and prev_p1_hp is not None:
            d = p1_hp - prev_p1_hp
            if d < 0:
                p1_hp_loss += -d

        if isinstance(p2_hp, (int, float)) and prev_p2_hp is not None:
            d = p2_hp - prev_p2_hp
            if d < 0:
                p2_hp_loss += -d

        prev_p1_hp = p1_hp
        prev_p2_hp = p2_hp

        # --- Status turns counting (exclude neutral and faint) ---
        if p1_status not in ['nostatus', 'fnt']:
            features['p1_bad_status'] += 1

        if p2_status not in ['nostatus', 'fnt']:
            features['p2_bad_status'] += 1

        # --- Faint events counting ---
        if p1_status == 'fnt':
            features['p1_ko_count'] += 1

        if p2_status == 'fnt':
            features['p2_ko_count'] += 1

    # Convert accumulated HP loss from [0, 1] scale to percentage points
    features['p1_hp_loss'] = round(p1_hp_loss * 100, 2)
    features['p2_hp_loss'] = round(p2_hp_loss * 100, 2)

    return features

    

def create_features(data: list[dict], pokedex: Pokedex) -> pd.DataFrame:
    """
    Extracts a unified feature set for each battle in the dataset.
    """

    feature_list: list[dict] = []

    for battle in tqdm(data, desc="Extracting features"):
        features: dict = {}

        # Per-battle feature blocks
        features.update(speed_advantage(battle, pokedex))
        features.update(switch_dynamics_features(battle))
        features.update(status_advantages(battle))
        features.update(offensive_rate(battle))
        features.update(extract_status_features(battle))
        features.update(static_features(battle, pokedex))
        features.update(dynamic_features(battle))
        features.update(moves(battle))

        # Attach identifiers and target variable (if available)
        features['battle_id'] = battle.get('battle_id')
        if 'player_won' in battle:
            features['player_won'] = int(battle['player_won'])

        feature_list.append(features)

    return pd.DataFrame(feature_list).fillna(0)
