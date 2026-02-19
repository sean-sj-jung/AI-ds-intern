import pandas as pd
import numpy as np
import time
from pytabkit import Ensemble_TD_Classifier
from sklearn.metrics import accuracy_score

def main():
    # Load training data
    print("Loading training data...")
    train_df = pd.read_csv('data/train.csv')
    original_df = pd.read_csv('data/Original_Heart_Disease_Prediction.csv')

    # Combine original data and a larger sample of train data
    # Ensemble_TD is quite heavy, so let's use 20,000 to be safe on time.
    SAMPLE_SIZE = 20000 
    print(f"Sampling {SAMPLE_SIZE} rows from training data...")
    train_subset = train_df.sample(n=SAMPLE_SIZE, random_state=42)
    combined_df = pd.concat([original_df, train_subset], axis=0).reset_index(drop=True)

    # Prepare features and target
    X_train = combined_df.drop(['id', 'Heart Disease'], axis=1)
    y_train = combined_df['Heart Disease'].map({'Presence': 1, 'Absence': 0})

    # Load test data
    print("Loading test data...")
    test_df = pd.read_csv('data/test.csv')
    X_test = test_df.drop(['id'], axis=1)
    ids_test = test_df['id']

    # Initialize PyTabKit Ensemble classifier
    # n_cv=5 will perform bagging and potentially much better results
    print("Initializing PyTabKit Ensemble_TD_Classifier...")
    model = Ensemble_TD_Classifier(n_cv=5, random_state=42)

    # Train the model
    print("Training the model...")
    start_time = time.time()
    model.fit(X_train, y_train)
    print(f"Training completed in {time.time() - start_time:.2f} seconds.")

    # Make predictions (probabilities)
    print("Predicting test set...")
    start_time = time.time()
    probs = model.predict_proba(X_test)[:, 1]
    y_pred = (probs > 0.5).astype(int)
    print(f"Prediction completed in {time.time() - start_time:.2f} seconds.")

    # Save probabilities
    pd.DataFrame({'id': ids_test, 'prob': probs}).to_csv('test_probs_pytabkit_ensemble.csv', index=False)
    print("Saved test probabilities for pytabkit_ensemble")

    # Create submission file
    submission = pd.DataFrame({
        'id': ids_test,
        'Heart Disease': y_pred
    })
    submission.to_csv('submission_pytabkit_ensemble.csv', index=False)
    print("Submission saved to submission_pytabkit_ensemble.csv")

if __name__ == "__main__":
    main()
