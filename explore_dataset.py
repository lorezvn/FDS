from pprint import pprint
from src.data_loader import load_jsonl
from tqdm import tqdm
from collections import Counter

TRAIN_PATH = "Dataset/train.jsonl"
TEST_PATH = "Dataset/test.jsonl"

print(f"Loading data from '{TRAIN_PATH}'...")
train_data = load_jsonl(TRAIN_PATH)
print(f"Successfully loaded {len(train_data)} battles.")

print(f"Loading data from '{TEST_PATH}'...")
test_data = load_jsonl(TEST_PATH)
print(f"Successfully loaded {len(test_data)} battles.\n")


def extract_info(dataset, desc="Extracting infos"):
    set_effects = set()
    set_moves = set()
    set_status = set()
    priority_counter = Counter()

    for battle in tqdm(dataset, desc=desc):
        for turn in battle.get("battle_timeline", []):

            # -- Effects --
            p1_effects = turn.get("p1_pokemon_state", {}).get("effects", [])
            p2_effects = turn.get("p2_pokemon_state", {}).get("effects", [])

            for eff in p1_effects + p2_effects:
                set_effects.add(eff)

            # -- Moves --
            p1_move = turn.get("p1_move_details", {})
            p2_move = turn.get("p2_move_details", {})

            for move in [p1_move, p2_move]:
                if not move:
                    continue
                move_name = move.get("name")
                set_moves.add(move_name)

                priority = move.get("priority", 0)
                priority_counter.update([priority])

            # --- Status ---
            p1_status = turn.get("p1_pokemon_state", {}).get("status", "")
            p2_status = turn.get("p2_pokemon_state", {}).get("status", "")
            for status in [p1_status, p2_status]:
                set_status.add(status)

    return set_effects, set_moves, set_status, priority_counter


# Estrazione da train e test
train_effects, train_moves, train_status, train_priority = extract_info(train_data, "Train")
test_effects, test_moves, test_status, test_priority = extract_info(test_data, "Test")

# Unioni e differenze
all_effects = train_effects | test_effects
all_moves = train_moves | test_moves
all_status = train_status | test_status

new_effects = test_effects - train_effects
new_moves = test_moves - train_moves
new_status = test_status - train_status

# --- Report finale ---

print("\n===== EFFECTS =====")
print(f"# Train effects: {len(train_effects)}")
print(f"# Test effects: {len(test_effects)}")
print(f"# Unique effects overall: {len(all_effects)}")
print(f"# New effects in test: {len(new_effects)}\n")
pprint(all_effects)

print("\n===== MOVES =====")
print(f"# Train moves: {len(train_moves)}")
print(f"# Test moves: {len(test_moves)}")
print(f"# Unique moves overall: {len(all_moves)}")
print(f"# New moves in test: {len(new_moves)}\n")
pprint(all_moves)

print("\n===== STATUS =====")
print(f"# Train status: {len(train_status)}")
print(f"# Test status: {len(test_status)}")
print(f"# Unique status overall: {len(all_status)}")
print(f"# New status in test: {len(new_status)}\n")
pprint(all_status)

print("\n===== PRIORITY =====")
print(f"Train priority distribution: {dict(train_priority)}")
print(f"Test priority distribution: {dict(test_priority)}")