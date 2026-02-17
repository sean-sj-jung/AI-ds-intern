import pandas as pd
import numpy as np
import time
import catboost as cb
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

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

    # Identify categorical columns
    categorical_features = [
        'Sex', 'Chest pain type', 'FBS over 120', 'EKG results', 
        'Exercise angina', 'Slope of ST', 'Number of vessels fluro', 'Thallium'
    ]

    # Split for local validation
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.1, random_state=42, stratify=y)

    # Load test data
    print("Loading test data...")
    test_df = pd.read_csv('data/test.csv')
    X_test = test_df.drop(['id'], axis=1)
    ids_test = test_df['id']

    # Initialize CatBoost
    print("Training CatBoost model...")
    model = cb.CatBoostClassifier(
        iterations=1000,
        learning_rate=0.05,
        depth=6,
        l2_leaf_reg=3,
        random_seed=42,
        eval_metric='Accuracy',
        early_stopping_rounds=50,
        verbose=100
    )

    start_time = time.time()
    
    model.fit(
        X_train, y_train,
        cat_features=categorical_features,
        eval_set=(X_val, y_val)
    )
    
    print(f"Training completed in {time.time() - start_time:.2f} seconds.")

    # Validation accuracy
    val_preds = model.predict(X_val)
    val_acc = accuracy_score(y_val, val_preds)
    print(f"Validation Accuracy: {val_acc:.4f}")

    # Make predictions on test set
    print("Predicting test set...")
    test_preds = model.predict(X_test)

    # Create submission file
    submission = pd.DataFrame({
        'id': ids_test,
        'Heart Disease': test_preds
    })
    
    submission_path = 'submission_cb.csv'
    submission.to_csv(submission_path, index=False)
    print(f"Submission saved to {submission_path}")

if __name__ == "__main__":
    main()
