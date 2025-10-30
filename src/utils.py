from tqdm import tqdm
from .constants import stats

def create_pokedex(data):
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

def crit_rate(base_speed):
    rate = base_speed * 100 / 512
    return round(rate, 4)

def get_p2_team(battle, pokedex):
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


        
    
