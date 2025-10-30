from ..data_types import Pokedex

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
        
    
