from src.data_loader import load_jsonl, display_battle
from src.features import create_features, analyze_defense, vulnerability_score
from src.model import train_and_evaluate

def main():
    TRAIN_PATH = "Dataset/train.jsonl"
    TEST_PATH = "Dataset/test.jsonl"

    print(f"Loading data from '{TRAIN_PATH}'...")
    train_data = load_jsonl(TRAIN_PATH)
    print(f"Successfully loaded {len(train_data)} battles.")
    #display_battle(0, train_data)
    test_data = load_jsonl(TEST_PATH)

    print("\nProcessing training data...")
    train_df = create_features(train_data)
    print("\nProcessing test data...")
    test_df = create_features(test_data)

    print("\nTraining features preview:")
    print(train_df.head(5))

    train_and_evaluate(train_df, test_df)


if __name__ == '__main__':
    main()
