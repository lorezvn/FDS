from src.data_loader import load_jsonl, display_battle, create_pokedex
from src.features.battle_features import create_features
from src.model.train import train_and_evaluate
import src.config as config
import time

def timer(func):
    """
    A decorator that prints the total time of execution 
    of a func in minutes and seconds.
    """
    def wrapper(*args, **kwargs):
        start_time = time.perf_counter()
        result = func(*args, **kwargs) 
        end_time = time.perf_counter()
        duration_seconds = end_time - start_time
        minutes = int(duration_seconds // 60)
        seconds = int(duration_seconds % 60)
        print(f"\nTotal exec time: {minutes} min, {seconds} sec")
        return result
    return wrapper


class BattlePredictor:
    """
    Manages the entire loading, processing, and training pipeline.

    Attributes:
        pokedex (Pokedex): The Pokedex dictionary created by `process_pokedex`.
        train_df (pd.DataFrame): Training DataFrame with features.
        test_df (pd.DataFrame): Test DataFrame with features.
        n_features (int, optional): Number of top features to show in the analysis. Defaults to 15.
        tuning (bool, optional): Flag to enable model hyperparameter tuning. Defaults to False.
        display (bool, optional): Flag to enable printing of previews (e.g., battles). Defaults to False.
    """
    def __init__(self, n_features: int = 15, tuning: bool = False, display: bool = False):
        self.pokedex = None
        self.train_df = None
        self.test_df = None
        self.display = display
        self.tuning = tuning
        self.n_features = n_features
    
    def load_raw_data(self) -> tuple[list[dict], list[dict]]: 
        print(f"\nLoading training data from '{config.TRAIN_PATH}'...")
        train_data = load_jsonl(config.TRAIN_PATH)
        print(f"Successfully loaded {len(train_data)} battles.")

        print(f"\nLoading test data from '{config.TEST_PATH}'...")
        test_data = load_jsonl(config.TEST_PATH)
        print(f"Successfully loaded {len(test_data)} battles.")

        if self.display:
            print("\nDisplaying a battle from training data")
            display_battle(0, train_data)

        return train_data, test_data

    def process_pokedex(self, train_data: list[dict]) -> None:
        print("\nProcessing pokemons...")
        self.pokedex = create_pokedex(train_data)

    def engineer_features(self, train_data: list[dict], test_data: list[dict]) -> None:
        if not self.pokedex:
            raise ValueError("Pokedex not loaded!")
        
        print("\nProcessing training data...")
        self.train_df = create_features(train_data, self.pokedex)

        print("\nProcessing test data...")
        self.test_df = create_features(test_data, self.pokedex)

        if self.display:
            print("\nTraining features preview:")
            print(self.train_df.head(5))

    @timer
    def run(self) -> None:

        # Loading data
        train_data, test_data = self.load_raw_data()

        # Pokedex
        self.process_pokedex(train_data)

        # Feature engineering
        self.engineer_features(train_data, test_data)

        # Training and evaluation
        print("\n=======================================================")
        print("TRAINING & EVALUATION", end=" ")
        train_and_evaluate(self.train_df, self.test_df, top_n=self.n_features, tuning=self.tuning)
        print("=======================================================")


if __name__ == '__main__':
    predictor = BattlePredictor(n_features=10)
    predictor.run()
