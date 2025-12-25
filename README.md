# Pokémon Battles Prediction
A **machine learning** system that predicts the **winner** of a **_Pokémon_ battle** based on a dataset of structured real battles.

- Lorenzo Zanda (2006432)
- Davide Vittucci (1903954)
- Paolo Marchetti (1986485)

## Feature Engineering
Battles are transformed into a set of **features** describing:
- Turn-by-turn decision patterns
- Damage and HP progression
- Status effects and type matchups
- Team mean stats
- Endgame conditions

## Modeling 
We train and evaluate a **Stacking Classifier** with:
- Logistic Regression
- Random Forest
- LightGBM
- SVC

**K-fold cross validation** (k=5) was used for a correct evaluation.

## Results 
| Model | Mean CV Accuracy |
|-------|------------------|
| Logistic Regression | 0.8492 |
| LightGBM | 0.8434 |
| SVC | 0.8456 |
| Random Forest | 0.8357 |
| **StackingClassifier** | **0.8505** |

## Project structure

* `src/` 
	* `src/features/` cointains all features extraction related files.
	* `src/model/` contains files for defining and training the model.
	* `src/data_loader.py` functions for loading raw battle data.
	* `src/data_types.py` defines the `Pokedex` datatype.
	* `src/config.py` global config settings.
* `notebooks/` contains the main kaggle notebook.
* `report/` contains the report in eng.
* `main.py` defines the `BattlePredictor` class that manages the training pipeline.


## How to run
1. Install dependencies 
    ```python
    pip install -r requirements.txt
    ```
2. Run training
    ```python
    python main.py
    ```  
You can switch models or enable tuning for the **StackingClassifier**.

## Credits
Project developed for the **FDS 2025-2026 Challenge**.
