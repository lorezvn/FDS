from ..data_types import Pokedex
from .constants import types_dict

def crit_rate(base_speed: int) -> float:
    rate = base_speed * 100 / 512
    return round(rate, 4)

def get_p2_team(battle: dict, pokedex: Pokedex):
    p2_team = []
    seen = set()

    lead_name = battle.get('p2_lead_details', {}).get('name')
    if lead_name and lead_name in pokedex:
        entry = {'name': lead_name, **pokedex[lead_name]}
        p2_team.append(entry)
        seen.add(lead_name)
        
    for turn in battle.get('battle_timeline', []):
        pokemon_name = turn.get('p2_pokemon_state', {}).get('name')
        if pokemon_name not in seen and pokemon_name in pokedex:
            entry = {'name': pokemon_name, **pokedex[pokemon_name]}
            p2_team.append(entry)
            seen.add(pokemon_name)

    return p2_team

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

def analyze_defense(pokemon: dict) -> dict:
    """
    Computes the defensive multipliers of a Pokémon against all attacking types.

    For each attacking type, the function determines how much damage the Pokémon
    would receive based on its own defensive typings. The calculation follows 
    standard type effectiveness rules:

      - Immunity → 0x
      - Resistance → 0.5x (stacking to 0.25x if double resistant)
      - Weakness → 2x (stacking to 4x if double weak)

    Returned dictionary groups attacking types into the following categories:
        - "0x":    Types that deal no damage (immunity)
        - "1/4x":  Types that deal quarter damage
        - "1/2x":  Types that deal half damage
        - "2x":    Types that deal double damage
        - "4x":    Types that deal quadruple damage
    """

    # Filter out invalid placeholder types
    def_types = [t for t in pokemon.get("types", []) if t != "notype"]

    multipliers = {}

    # Compute final defensive multiplier for each attacking type
    for atk_type in types_dict:

        # Check if any defensive type grants immunity
        if any(atk_type in types_dict[def_t]["immune"] for def_t in def_types):
            multiplier = 0.0
        else:
            multiplier = 1.0
            # Apply weaknesses and resistances
            for def_t in def_types:
                if atk_type in types_dict[def_t]["weakness"]:
                    multiplier *= 2.0
                if atk_type in types_dict[def_t]["resistence"]:
                    multiplier *= 0.5

        multipliers[atk_type] = multiplier

    # Group attacking types by their final multipliers
    mult_list = {"0x": [], "1/4x": [], "1/2x": [], "2x": [], "4x": []}

    for atk, m in multipliers.items():
        if m == 0.0:
            mult_list["0x"].append(atk)
        elif m == 0.25:
            mult_list["1/4x"].append(atk)
        elif m == 0.5:
            mult_list["1/2x"].append(atk)
        elif m == 2.0:
            mult_list["2x"].append(atk)
        elif m == 4.0:
            mult_list["4x"].append(atk)

    return mult_list
        
    
