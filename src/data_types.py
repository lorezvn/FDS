from typing import Dict, List, TypedDict, Tuple

class PokemonStats(TypedDict):
    base_hp: int
    base_atk: int
    base_def: int
    base_spa: int
    base_spd: int
    base_spe: int
    types: List[str]

Pokedex = Dict[str, PokemonStats]