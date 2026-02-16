import pandas as pd
import numpy as np
import time
from pytabkit import RealMLP_TD_Classifier
from sklearn.metrics import accuracy_score

def main():
    # Load training data
    print("Loading training data...")
    train_df = pd.read_csv('data/train.csv')
    original_df = pd.read_csv('data/Original_Heart_Disease_Prediction.csv')

    # Combine original data and train data
    # pytabkit is generally more scalable than TabPFN, but 600k might still be large for a single fit.
    # Let's try using a larger subset or the full data if possible.
    # Final run with larger training sample and full test set
    SAMPLE_SIZE = 30000 
    print(f"Sampling {SAMPLE_SIZE} rows from training data...")
    train_subset = train_df.sample(n=SAMPLE_SIZE, random_state=42)
    combined_df = pd.concat([original_df, train_subset], axis=0).reset_index(drop=True)

    # Prepare features and target
    # map Presence/Absence to 1/0
    X_train = combined_df.drop(['id', 'Heart Disease'], axis=1)
    y_train = combined_df['Heart Disease'].map({'Presence': 1, 'Absence': 0})

    # Load test data
    print("Loading test data...")
    test_df = pd.read_csv('data/test.csv')
    X_test = test_df.drop(['id'], axis=1)
    ids_test = test_df['id']

    # Initialize PyTabKit classifier
    # n_cv=1 to start for speed, can increase to 5 for better accuracy (bagging)
    print("Initializing PyTabKit (RealMLP_TD_Classifier)...")
    model = RealMLP_TD_Classifier(n_cv=1, random_state=42)

    # Train the model
    print("Training the model...")
    start_time = time.time()
    model.fit(X_train, y_train)
    print(f"Training completed in {time.time() - start_time:.2f} seconds.")

    # Make predictions
    print("Predicting test set...")
    start_time = time.time()
    # pytabkit predict can handle large batches typically
    y_pred = model.predict(X_test)
    print(f"Prediction completed in {time.time() - start_time:.2f} seconds.")

    # Create submission file
    submission = pd.DataFrame({
        'id': ids_test,
        'Heart Disease': y_pred
    })
    
    submission_path = 'submission_pytabkit.csv'
    submission.to_csv(submission_path, index=False)
    print(f"Submission saved to {submission_path}")

    # Optional: Quick check on training accuracy
    train_preds = model.predict(X_train)
    train_acc = accuracy_score(y_train, train_preds)
    print(f"Training Accuracy: {train_acc:.4f}")

if __name__ == "__main__":
    main()
