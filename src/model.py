import os
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, GridSearchCV, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.model_selection import cross_validate

def get_best_model(X_train, y_train):

    pipe = make_pipeline(
        StandardScaler(), 
        SelectKBest(score_func=f_classif),
        LogisticRegression(max_iter=1000, random_state=42)
    )

    # Define the parameter grid to search
    param_grid = [
        {
            'selectkbest__k': [10, 20, 30, 'all'],
            'logisticregression__C': [0.01, 0.1, 1, 10, 100],
            'logisticregression__penalty': ['l2'],
            'logisticregression__solver': ['lbfgs', 'newton-cg', 'newton-cholesky', 'sag']
        }, 
    ]

    # Create the GridSearchCV object
    grid_logreg = GridSearchCV(
        estimator=pipe,
        param_grid=param_grid,
        scoring='accuracy',
        n_jobs=4,        
        cv=5,            # 5-fold cross-validation, more on this later
        refit=True,      # retrain the best model on the full training set
        return_train_score=True
    )

    # Fit the GridSearchCV object on the training data
    grid_logreg.fit(X_train, y_train)

    # Print the best accuracy score found during grid search
    best_score = grid_logreg.best_score_
    print(f"\nBest accuracy score: {best_score:.4f}")

    # Extract the best hyperparameter combination
    best_params = grid_logreg.best_params_
    print(f"Best hyperparameters: {best_params}")

    return grid_logreg.best_estimator_


def cross_validate(model, X_train, y_train): 

    kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    # split()  method generate indices to split data into training and test set.
    print()
    for count, (train_index, test_index) in enumerate(kf.split(X_train, y_train)):
        print(f"Fold: {count}, Train set: {len(train_index)}, Test set:{len(test_index)}")

    # Cross validation
    score = cross_val_score(model, X_train, y_train, cv=kf, scoring="accuracy")
    print(f"Scores for each fold are: {score}")
    print(f"Cross-val mean: {score.mean():.4f} ± {score.std():.4f}")


def feature_correlations(model, X_train, top_n=10):

    output_dir = "outputs/plots"

    # Top features
    coefs = model.named_steps['logisticregression'].coef_[0]
    features = X_train.columns
    top_idx = np.argsort(np.abs(coefs))[::-1][:top_n]
    top_features = features[top_idx]

    print(f"\nTop {top_n} features by absolute weight:")
    for i, (f, c) in enumerate(zip(top_features, coefs[top_idx])):
        print(f"{i+1}) {f}: {c:.3f}")

    # Heatmap
    corr = X_train[top_features].corr()
    plt.figure(figsize=(12, 10))
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", center=0, vmin=-1, vmax=1)
    plt.title(f"Top {top_n} Feature Correlation Heatmap")

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    file_path = os.path.join(output_dir, f"top_{top_n}_features_heatmap.png")
    plt.savefig(file_path, dpi=300, bbox_inches="tight")

    print(f"\n'top_{top_n}_features_heatmap.png' file created successfully!")


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

    feature_correlations(model, X_train, top_n=15)

    generate_submission(model, X_train, y_train, X_test, test_df)