# immune = moves that deal 0x to my pokemons
# weakness = moves that deal 2x to my pokemons
# resistence = moves that deal 1/2x to my pokemons

types_dict = {
    "normal": {
        "immune": ["ghost"],
        "weakness": ["fighting"],
        "resistence": []
    },
    "fire": {
        "immune": [],
        "weakness": ["water", "ground", "rock"],
        "resistence": ["fire", "grass", "bug"]
    },
    "water": {
        "immune": [],
        "weakness": ["electric", "grass"],
        "resistence": ["fire", "water", "ice"]
    },
    "electric": {
        "immune": [],
        "weakness": ["ground"],
        "resistence": ["electric", "flying"]
    },
    "grass": {
        "immune": [],
        "weakness": ["fire", "ice", "poison", "flying", "bug"],
        "resistence": ["water", "electric", "grass", "ground"]
    },
    "ice": {
        "immune": [],
        "weakness": ["fire", "fighting", "rock"],
        "resistence": ["ice"]
    },
    "fighting": {
        "immune": [],
        "weakness": ["flying", "psychic"],
        "resistence": ["bug", "rock"]
    },
    "poison": {
        "immune": [],
        "weakness": ["ground", "psychic", "bug"],
        "resistence": ["fighting", "poison", "grass"]
    },
    "ground": {
        "immune": ["electric"],
        "weakness": ["water", "grass", "ice"],
        "resistence": ["poison", "rock"]
    },
    "flying": {
        "immune": ["ground"],
        "weakness": ["electric", "ice", "rock"],
        "resistence": ["grass", "fighting", "bug"]
    },
    "psychic": {
        "immune": ["ghost"],
        "weakness": ["bug"],
        "resistence": ["fighting", "psychic"]
    },
    "bug": {
        "immune": [],
        "weakness": ["fire", "flying", "rock", "poison"],
        "resistence": ["grass", "fighting", "ground"]
    },
    "rock": {
        "immune": [],
        "weakness": ["water", "grass", "fighting", "ground"],
        "resistence": ["normal", "fire", "poison", "flying"]
    },
    "ghost": {
        "immune": ["normal", "fighting"],
        "weakness": ["ghost"],
        "resistence": ["poison", "bug"]
    },
    "dragon": {
        "immune": [],
        "weakness": ["ice", "dragon"],
        "resistence": ["fire", "water", "electric", "grass"]
    }
}