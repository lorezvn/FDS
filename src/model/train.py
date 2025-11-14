import pandas as pd
from sklearn.model_selection import StratifiedKFold, cross_val_score
from .model import get_final_model
import src.config as config


def cross_validate(model, X_train, y_train): 

    print("\n--- K-Fold Cross Validation ---")
    kf = StratifiedKFold(n_splits=config.CV_FOLDS, shuffle=True, random_state=config.RANDOM_STATE)

    #for count, (train_index, test_index) in enumerate(kf.split(X_train, y_train)):
        #print(f"Fold: {count}, Train set: {len(train_index)}, Test set:{len(test_index)}")

    print("Estimators evaluation:")
    for name, est in model.estimators:
        score = cross_val_score(est, X_train, y_train, cv=kf, scoring="accuracy", n_jobs=config.N_JOBS)
        print(f"  > {name:>6}: mean acc = {score.mean():.4f} (+/- {score.std():.4f})")

    # Cross validation
    print("\nStacked model evaluation:")
    score = cross_val_score(model, X_train, y_train, cv=kf, scoring="accuracy", n_jobs=config.N_JOBS)
    print(f"  > Fold scores: {score}")
    print(f"  > Cross-val mean: {score.mean():.4f} (+/- {score.std():.4f})")


def generate_submission(model, X_test, test_df):

    # Make predictions on the test data
    test_predictions = model.predict(X_test)

    # Create the submission DataFrame
    submission_df = pd.DataFrame({
        'battle_id': test_df['battle_id'],
        'player_won': test_predictions
    })

    # Save the DataFrame to a .csv file
    submission_df.to_csv(config.SUBMISSION_PATH, index=False)
    print("'submission.csv' file created successfully!")


def train_and_evaluate(train_df, test_df, top_n, tuning):

    # Only allowed features 
    features = [col for col in train_df.columns if col not in ['battle_id', 'player_won'] + config.FEATURES_TO_DROP]

    print(f"(with {len(features)} features)")

    X_train = train_df[features]
    y_train = train_df['player_won']

    X_test = test_df[features]

    model = get_final_model(X_train, y_train, tuning)

    cross_validate(model, X_train, y_train)

    print("\nTraining model...")
    model.fit(X_train, y_train)
    print("Model trained!")

    generate_submission(model, X_test, test_df)