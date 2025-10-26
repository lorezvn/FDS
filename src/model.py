import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline


def cross_validate(model, X_train, y_train): 

    kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    # split()  method generate indices to split data into training and test set.
    print()
    for count, (train_index, test_index) in enumerate(kf.split(X_train, y_train)):
        print(f'Fold: {count}, Train set: {len(train_index)}, Test set:{len(test_index)}')

    # Cross validation
    score = cross_val_score(model, X_train, y_train, cv=kf, scoring="accuracy")
    print(f'Scores for each fold are: {score}')
    print(f'Average score: {"{:.2f}".format(score.mean())}')


def generate_submission(model, X_train, y_train, X_test, test_df):

    # Train model
    print("\nTraining model...")
    model.fit(X_train, y_train)

    # Make predictions on the test data
    print("\nGenerating predictions on the test set...")
    test_predictions = model.predict(X_test)

    # Create the submission DataFrame
    submission_df = pd.DataFrame({
        'battle_id': test_df['battle_id'],
        'player_won': test_predictions
    })

    # Save the DataFrame to a .csv file
    submission_df.to_csv('outputs/submission.csv', index=False)
    print("\n'submission.csv' file created successfully!")


def train_and_evaluate(train_df, test_df):
    features = [col for col in train_df.columns if col not in ['battle_id', 'player_won']]

    X_train = train_df[features]
    y_train = train_df['player_won']

    X_test = test_df[features]

    # Create pipeline
    model = make_pipeline(StandardScaler(), 
                        LogisticRegression(random_state=42, max_iter=1000))

    cross_validate(model, X_train, y_train)

    generate_submission(model, X_train, y_train, X_test, test_df)