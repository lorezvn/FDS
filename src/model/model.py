from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier, StackingClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
import src.config as config

def get_tuned_params(name, estimator, X_train, y_train):

    param_grid = {} 
    
    match name:
        case 'rf': 
           param_grid = {
            'n_estimators': [300, 400, 500, 600],
            'max_depth': [8, 10, 12, 15], 
            'max_features': ['sqrt', 'log2']
        } 
        case 'lgbm': 
            param_grid = {
                'n_estimators': [200, 400, 600],
                'learning_rate': [0.01, 0.05, 0.1],
                'num_leaves': [20, 31, 40], 
                'max_depth': [5, 7, 10, 15]
            }
        case 'xgb': 
            param_grid = {
            'n_estimators': [300, 500, 700],
            'learning_rate': [0.01, 0.05, 0.1],
            'max_depth': [3, 5, 7]
        }
        case 'svc':
            param_grid = {
                'svc__C': [0.1, 1, 10, 100],      
                'svc__gamma': ['scale', 'auto', 0.01, 0.1] 
            }
        case _:
            print(f"\nNo tuning for {name}. Using default")
            return {}


    print(f"\nTuning for {name}...")
    
    random_search = RandomizedSearchCV(  
        estimator=estimator,
        param_distributions=param_grid,  
        n_iter=30,                       
        scoring='accuracy',
        n_jobs=config.N_JOBS, 
        cv=config.CV_FOLDS,
        verbose=1,
        refit=False,
        random_state=config.RANDOM_STATE
    )
    
    random_search.fit(X_train, y_train)

    best_score = random_search.best_score_
    best_params = random_search.best_params_
    
    print(f"  > Best score: {best_score:.4f}")
    print(f"  > Best hyperparameters: {best_params}")
    
    if name == 'svc':
        best_params = {key.split('__')[1]: val for key, val in best_params.items()}
        
    return best_params


def get_final_model(X_train, y_train, tuning=False):
    estimators = [
        ('lr', make_pipeline(StandardScaler(), LogisticRegression(solver='lbfgs', max_iter=1000, random_state=config.RANDOM_STATE))),
        ('rf', RandomForestClassifier(n_estimators=400, max_features='sqrt', max_depth=15, random_state=config.RANDOM_STATE, n_jobs=config.N_JOBS)),
        ('lgbm', LGBMClassifier(num_leaves=20, n_estimators=200, max_depth=7, learning_rate=0.05, random_state=config.RANDOM_STATE, n_jobs=config.N_JOBS, verbose=-1)),
        #('xgb', XGBClassifier(eval_metric='logloss', random_state=config.RANDOM_STATE)),
        ('svc', make_pipeline(StandardScaler(), SVC(gamma='scale', C=1, probability=False, random_state=config.RANDOM_STATE)))
    ]

    if tuning:
        
        print("\n--- Tuning ---")
        tuned_estimators = []

        for name, estimator in estimators:

            best_params = get_tuned_params(name, estimator, X_train, y_train)
            
            if name == 'lr':
                best_params.update({'max_iter': 1000, 'solver': 'lbfgs', 'random_state': config.RANDOM_STATE})
                tuned_estimators.append(('lr', make_pipeline(StandardScaler(), LogisticRegression(**best_params))))
            elif name == 'rf':
                best_params.update({'random_state': config.RANDOM_STATE, 'n_jobs': config.N_JOBS})
                tuned_estimators.append(('rf', RandomForestClassifier(**best_params)))
            elif name == 'lgbm':
                best_params.update({'random_state': config.RANDOM_STATE, 'n_jobs': config.N_JOBS, 'verbose': -1})
                tuned_estimators.append(('lgbm', LGBMClassifier(**best_params)))
            elif name == 'xgb':
                best_params.update({'random_state': config.RANDOM_STATE, 'eval_metric': 'logloss'})
                tuned_estimators.append(('xgb', XGBClassifier(**best_params)))
            elif name == 'svc':
                best_params.update({'probability': False, 'random_state': config.RANDOM_STATE})
                tuned_estimators.append(('svc', make_pipeline(StandardScaler(), SVC(**best_params))))
        
        estimators = tuned_estimators

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
