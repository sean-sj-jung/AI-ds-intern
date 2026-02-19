import pandas as pd
import numpy as np
import time
import xgboost as xgb
import lightgbm as lgb
import catboost as cb
from sklearn.metrics import accuracy_score
from sklearn.model_selection import StratifiedKFold
import os

def main():
    # Load training data
    print("Loading training data...")
    train_df = pd.read_csv('data/train.csv')
    original_df = pd.read_csv('data/Original_Heart_Disease_Prediction.csv')

    # Combine original data and full train data
    print("Combining datasets...")
    combined_df = pd.concat([original_df, train_df], axis=0).reset_index(drop=True)

    # Prepare features and target
    X = combined_df.drop(['id', 'Heart Disease'], axis=1)
    y = combined_df['Heart Disease'].map({'Presence': 1, 'Absence': 0})

    # Load test data
    print("Loading test data...")
    test_df = pd.read_csv('data/test.csv')
    X_test = test_df.drop(['id'], axis=1)
    ids_test = test_df['id']

    # Identify categorical columns
    categorical_features = [
        'Sex', 'Chest pain type', 'FBS over 120', 'EKG results', 
        'Exercise angina', 'Slope of ST', 'Number of vessels fluro', 'Thallium'
    ]

    # K-Fold Strategy for OOF and Test Averaging
    N_SPLITS = 5
    skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=42)
    
    # Store test predictions (probabilities)
    test_preds_total = {
        'xgb': np.zeros(len(X_test)),
        'lgb': np.zeros(len(X_test)),
        'cb': np.zeros(len(X_test))
    }

    print(f"Starting {N_SPLITS}-fold training...")
    
    for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
        print(f"\n--- Fold {fold+1}/{N_SPLITS} ---")
        X_tr, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_tr, y_val = y.iloc[train_idx], y.iloc[val_idx]

        # 1. XGBoost with better hyperparams
        print("Training XGBoost...")
        xgb_model = xgb.XGBClassifier(
            n_estimators=2000,
            learning_rate=0.03,
            max_depth=7,
            min_child_weight=1,
            subsample=0.8,
            colsample_bytree=0.8,
            tree_method='hist',
            random_state=42,
            early_stopping_rounds=100,
            eval_metric='error'
        )
        xgb_model.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], verbose=False)
        test_preds_total['xgb'] += xgb_model.predict_proba(X_test)[:, 1] / N_SPLITS

        # 2. LightGBM with better hyperparams
        print("Training LightGBM...")
        lgb_model = lgb.LGBMClassifier(
            n_estimators=2000,
            learning_rate=0.03,
            num_leaves=63,
            feature_fraction=0.8,
            bagging_fraction=0.8,
            bagging_freq=5,
            random_state=42,
            verbosity=-1
        )
        lgb_model.fit(
            X_tr, y_tr, 
            eval_set=[(X_val, y_val)],
            callbacks=[lgb.early_stopping(100), lgb.log_evaluation(0)]
        )
        test_preds_total['lgb'] += lgb_model.predict_proba(X_test)[:, 1] / N_SPLITS

        # 3. CatBoost with better hyperparams
        print("Training CatBoost...")
        cb_model = cb.CatBoostClassifier(
            iterations=2000,
            learning_rate=0.03,
            depth=7,
            l2_leaf_reg=5,
            random_seed=42,
            eval_metric='Accuracy',
            early_stopping_rounds=100,
            verbose=False
        )
        cb_model.fit(X_tr, y_tr, cat_features=categorical_features, eval_set=(X_val, y_val))
        test_preds_total['cb'] += cb_model.predict_proba(X_test)[:, 1] / N_SPLITS

    # Save individual probability files for ensembling later
    for name, preds in test_preds_total.items():
        pd.DataFrame({'id': ids_test, 'prob': preds}).to_csv(f'test_probs_{name}.csv', index=False)
        print(f"Saved test probabilities for {name}")

    # Simple Average Ensemble of these 3
    final_prob = (test_preds_total['xgb'] + test_preds_total['lgb'] + test_preds_total['cb']) / 3
    final_preds = (final_prob > 0.5).astype(int)

    submission = pd.DataFrame({'id': ids_test, 'Heart Disease': final_preds})
    submission.to_csv('submission_ensemble_gbdt.csv', index=False)
    print("Ensemble submission saved as submission_ensemble_gbdt.csv")

if __name__ == "__main__":
    main()
