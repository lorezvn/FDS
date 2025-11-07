import os
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, GridSearchCV, RandomizedSearchCV, cross_val_score
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, StackingClassifier
from xgboost import XGBClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.model_selection import cross_validate
from datetime import datetime
import config

# FILE SOLO DI RIFERIMENTO DA NON CONSIDERARE

def get_best_model(X_train, y_train):

    pipe = make_pipeline(
        StandardScaler(), 
        SelectKBest(score_func=f_classif),
        LogisticRegression(max_iter=1000, random_state=config.RANDOM_STATE)
    )

    # Define the parameter grid to search
    param_grid = [
        {
            'selectkbest__k': [10, 20, 30, 'all'],
            'logisticregression__C': [0.01, 0.1, 1, 10, 100],
            'logisticregression__penalty': ['l2'],
            'logisticregression__solver': ['lbfgs', 'sag']
        }, 
    ]

    # Create the GridSearchCV object
    grid_logreg = GridSearchCV(
        estimator=pipe,
        param_grid=param_grid,
        scoring='accuracy',
        n_jobs=config.N_JOBS,        
        cv=config.CV_FOLDS,            
        refit=True,      
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

# -----------------------------------------------------------------
# 1. FUNZIONE HELPER "SMART": _get_tuned_params
# (Fa il tuning di un singolo modello)
# -----------------------------------------------------------------
def _get_tuned_params(name, estimator, X_train, y_train):
    """
    Funzione "intelligente" che sa quale griglia di parametri usare
    per ogni modello ed esegue il tuning.
    """
    
    param_grid = {} # Griglia vuota di default
    
    # Definiamo la griglia giusta in base al NOME
    if name == 'rf':
        param_grid = {
            'n_estimators': [300, 400, 500, 600],
            'max_depth': [8, 10, 12, 15]
        }
    elif name == 'xgb':
        param_grid = {
            'n_estimators': [300, 500, 700],
            'learning_rate': [0.01, 0.05, 0.1],
            'max_depth': [3, 5, 7]
        }
    elif name == 'svc':
        param_grid = {
            'svc__C': [0.1, 1, 10],      
            'svc__gamma': ['scale', 'auto'] 
        }
    
    # Se la griglia è vuota (es. per 'lr'), non facciamo tuning
    if not param_grid:
        print(f"\nNessun tuning per: {name}. Uso i default.")
        return {} # Ritorna parametri vuoti

    # Esegui il tuning
    print(f"\nInizio tuning per: {name}...")
    
    random_search = RandomizedSearchCV(  # <<< MODIFICA
        estimator=estimator,
        param_distributions=param_grid,  # <<< MODIFICA
        n_iter=30,                       # <<< Prova 30 combinazioni a caso
        scoring='accuracy',
        n_jobs=-1, # Usa tutti i core
        cv=config.CV_FOLDS,
        verbose=1,
        refit=False,
        random_state=config.RANDOM_STATE
    )
    
    random_search.fit(X_train, y_train)
    
    print(f"--- Risultati per: {name} ---")
    print(f"Miglior score: {random_search.best_score_:.4f}")
    print(f"Migliori iperparametri: {random_search.best_params_}")
    print("----------------------------------")
    
    # Pulisce i nomi dei parametri se necessario (per SVC)
    best_params = random_search.best_params_
    if name == 'svc':
        best_params = {key.split('__')[1]: val for key, val in best_params.items()}
        
    return best_params

# -----------------------------------------------------------------
# 2. FUNZIONE 'MASTER': get_new_model
# (Come l'hai chiesta: fa tutto lei)
# -----------------------------------------------------------------
def get_stacked_model(X_train, y_train):

    # 1. Definiamo la "lista" dei modelli base grezzi
    estimators_to_tune = [
        ('lr', make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000, solver='lbfgs', random_state=config.RANDOM_STATE))),
        ('rf', RandomForestClassifier(random_state=config.RANDOM_STATE, n_jobs=config.N_JOBS)),
        ('xgb', XGBClassifier(random_state=config.RANDOM_STATE)),
        ('svc', make_pipeline(StandardScaler(), SVC(probability=True, random_state=config.RANDOM_STATE)))
    ]
    
    tuned_estimators_list = [] 

    # 2. Il loop che esegue il tuning
    for name, estimator in estimators_to_tune:

        best_params = _get_tuned_params(name, estimator, X_train, y_train)
        
        # 4. Ricostruisce l'estimator con i parametri trovati
        if name == 'lr':
            # 'lr' non ha tuning, usiamo i parametri di default
            best_params.update({'max_iter': 1000, 'solver': 'lbfgs', 'random_state': config.RANDOM_STATE})
            tuned_estimators_list.append(('lr', make_pipeline(StandardScaler(), LogisticRegression(**best_params))))
        elif name == 'rf':
            best_params.update({'random_state': config.RANDOM_STATE, 'n_jobs': config.N_JOBS})
            tuned_estimators_list.append(('rf', RandomForestClassifier(**best_params)))
        elif name == 'xgb':
            best_params.update({'random_state': config.RANDOM_STATE})
            tuned_estimators_list.append(('xgb', XGBClassifier(**best_params)))
        elif name == 'svc':
            best_params.update({'probability': True, 'random_state': config.RANDOM_STATE})
            tuned_estimators_list.append(('svc', make_pipeline(StandardScaler(), SVC(**best_params))))
    
    # Costruiamo uno stack *provvisorio* per il tuning
    stacking_model = StackingClassifier(
        estimators=tuned_estimators_list, # Usa i modelli base già ottimizzati
        final_estimator=LogisticRegression(penalty='l2', solver='lbfgs', max_iter=1000, random_state=config.RANDOM_STATE),
        cv=config.CV_FOLDS,  
        n_jobs=config.N_JOBS, 
        passthrough=False 
    )

    print("\nModello Stacking finale costruito e ottimizzato. Pronto.")
    return stacking_model


def get_final_model():
    estimators = [
        ('lr', make_pipeline(StandardScaler(), LogisticRegression(solver='lbfgs', max_iter=1000, random_state=config.RANDOM_STATE))),
        ('rf', RandomForestClassifier(n_estimators=400, max_depth=8, random_state=config.RANDOM_STATE, n_jobs=config.N_JOBS)),
        ('xgb', XGBClassifier(eval_metric='logloss', random_state=config.RANDOM_STATE)),
        ('svc', make_pipeline(StandardScaler(), SVC(probability=True, random_state=config.RANDOM_STATE)))
    ]

    final_estimator = make_pipeline(
        StandardScaler(),
        LogisticRegression(C=10, penalty='l2', solver='lbfgs', max_iter=1000, random_state=config.RANDOM_STATE)
    )

    stacking_model = StackingClassifier(
        estimators=estimators,
        final_estimator=final_estimator,
        cv=config.CV_FOLDS,  
        n_jobs=config.N_JOBS, 
        passthrough=True
    )

    return stacking_model


def cross_validate(model, X_train, y_train): 

    print("\nStarting K-Fold Cross Validation...")
    kf = StratifiedKFold(n_splits=config.CV_FOLDS, shuffle=True, random_state=config.RANDOM_STATE)

    #for count, (train_index, test_index) in enumerate(kf.split(X_train, y_train)):
        #print(f"Fold: {count}, Train set: {len(train_index)}, Test set:{len(test_index)}")

    print("\nEvaluating base learners individually...")
    for name, est in model.estimators:
        score = cross_val_score(est, X_train, y_train, cv=kf, scoring="accuracy", n_jobs=-1)
        print(f"  {name:>6}: mean acc = {score.mean():.4f} +/- {score.std():.4f}")

    # Cross validation
    print("\nEvaluating stacked model...")
    score = cross_val_score(model, X_train, y_train, cv=kf, scoring="accuracy", n_jobs=-1)
    print(f"\nStacked model scores: {score}")
    print(f"Cross-val mean: {score.mean():.4f} +/- {score.std():.4f}")


def feature_correlations(model, X_train, top_n=10):

    output_dir = "outputs/plots"

    # Top features
    rf = model.named_estimators_['rf']
    importances = rf.feature_importances_
    features = X_train.columns
    top_idx = np.argsort(importances)[::-1][:top_n]
    top_features = features[top_idx]

    print(f"\nTop {top_n} features by absolute weight:")
    for i, (f, c) in enumerate(zip(top_features, importances[top_idx])):
        print(f"{i+1}) {f}: {c:.3f}")

    # Heatmap
    corr = X_train[top_features].corr()
    plt.figure(figsize=(12, 10))
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", center=0, vmin=-1, vmax=1)
    plt.title(f"Top {top_n} Feature Correlation Heatmap")

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    file_path = os.path.join(output_dir, f"{timestamp}_top_{top_n}.png")
    plt.savefig(file_path, dpi=300, bbox_inches="tight")

    print(f"\n'{timestamp}_top_{top_n}.png' file created successfully!")


def generate_submission(model, X_test, test_df):

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

    model = get_stacked_model(X_train, y_train)

    cross_validate(model, X_train, y_train)

    #print("\nTraining model...")
    #model.fit(X_train, y_train)

    feature_correlations(model, X_train, top_n=15)

    generate_submission(model, X_test, test_df)