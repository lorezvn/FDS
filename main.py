<<<<<<< Updated upstream
from src.data_loader import load_jsonl, display_battle, create_pokedex
=======
from src.data_loader import load_jsonl, display_battle
>>>>>>> Stashed changes
from src.features import create_features, speed_adv_rate
from src.model import train_and_evaluate
from src.utils import get_p2_team

def main():
    TRAIN_PATH = "Dataset/train.jsonl"
    TEST_PATH = "Dataset/test.jsonl"

    print(f"Loading data from '{TRAIN_PATH}'...")
    train_data = load_jsonl(TRAIN_PATH)
    print(f"Successfully loaded {len(train_data)} battles.")
    #display_battle(0, train_data)
    test_data = load_jsonl(TEST_PATH)

    print("\nProcessing pokemons...")
    pokedex = create_pokedex(train_data)
<<<<<<< Updated upstream
    print(pokedex)
    return
=======
>>>>>>> Stashed changes
    
    print("\nProcessing training data...")
    train_df = create_features(train_data, pokedex)
    print("\nProcessing test data...")
    test_df = create_features(test_data, pokedex)

    print("\nTraining features preview:")
    print(train_df.head(5))

    train_and_evaluate(train_df, test_df)


if __name__ == '__main__':
    main()
