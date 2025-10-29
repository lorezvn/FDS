from tqdm import tqdm
from .constants import stats

def create_pokedex(data):
    pokedex = {}
    for battle in tqdm(data, desc="Create pokedex"):
        
        p1_team = battle.get('p1_team_details', [])
        p2_lead = battle.get('p2_lead_details')

        pokemon_list = p1_team + [p2_lead]

        for pokemon in pokemon_list:
            pokemon_name = pokemon.get('name')
            if pokemon_name not in pokedex:
                pokemon_stats = {f'base_{stat}': pokemon.get(f'base_{stat}') for stat in stats}
                
                pokedex[pokemon_name] = pokemon_stats

    return pokedex


        
    
