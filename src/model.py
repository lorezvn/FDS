import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import KFold, GridSearchCV, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

def get_best_model(X_train, y_train):

    pipe = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000, random_state=42))

    # Define the parameter grid to search
    param_grid = {
        'logisticregression__C': [0.01, 0.1, 1, 10],
        'logisticregression__penalty': ['l1', 'l2'],
        'logisticregression__solver': ['liblinear']
    }

    # Create the GridSearchCV object
    grid_logreg = GridSearchCV(
        estimator=pipe,
        param_grid=param_grid,
        scoring='accuracy',
        n_jobs=4,        # use 4 cores in parallel
        cv=5,            # 5-fold cross-validation, more on this later
        refit=True,      # retrain the best model on the full training set
        return_train_score=True
    )

    # Fit the GridSearchCV object on the training data
    grid_logreg.fit(X_train, y_train)

    # Print the best accuracy score found during grid search
    best_score = grid_logreg.best_score_
    print("Best accuracy score:", best_score)

    # Extract the best hyperparameter combination
    best_params = grid_logreg.best_params_
    print("\nBest hyperparameters:")
    print(best_params)

    return grid_logreg.best_estimator_


def cross_validate(model, X_train, y_train): 

    kf = KFold(n_splits=5, shuffle=True, random_state=42)

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

    # Get the best model
    model = get_best_model(X_train, y_train)

    cross_validate(model, X_train, y_train)

    generate_submission(model, X_train, y_train, X_test, test_df)