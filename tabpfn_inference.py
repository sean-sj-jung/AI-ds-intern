import pandas as pd
from tabpfn import TabPFNClassifier
import numpy as np
import time

def main():
    # Load training data
    print("Loading training data...")
    train_df = pd.read_csv('data/train.csv')
    original_df = pd.read_csv('data/Original_Heart_Disease_Prediction.csv')

    # Combine original data and a subset of train data
    # We use a subset to stay within TabPFN limits and ensure reasonable runtime
    # Reducing to 1000 to speed up inference for 270k test samples
    SAMPLE_SIZE = 730 
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

    # Initialize and fit TabPFN
    print("Initializing TabPFN...")
    from tabpfn.constants import ModelVersion
    clf = TabPFNClassifier.create_default_for_version(ModelVersion.V2, device='cpu', ignore_pretraining_limits=True)
    
    print("Fitting TabPFN (storing data)...")
    start_time = time.time()
    clf.fit(X_train, y_train)
    print(f"Fit completed in {time.time() - start_time:.2f} seconds.")

    # Predict in batches to avoid memory issues and track progress
    print("Predicting test set...")
    batch_size = 2000
    all_preds = []
    
    total_batches = (len(X_test) + batch_size - 1) // batch_size
    
    start_time = time.time()
    for i in range(0, len(X_test), batch_size):
        batch_X = X_test.iloc[i:i+batch_size]
        preds = clf.predict(batch_X)
        all_preds.extend(preds)
        
        # Report every batch to show progress
        elapsed = time.time() - start_time
        avg_time = elapsed / (i // batch_size + 1)
        remaining = avg_time * (total_batches - (i // batch_size + 1))
        print(f"Batch {i//batch_size + 1}/{total_batches} completed. "
              f"Elapsed: {elapsed:.2f}s, Estimated remaining: {remaining/60:.2f} min.")

    # Create submission file
    submission = pd.DataFrame({
        'id': ids_test,
        'Heart Disease': all_preds
    })
    
    # Map back to strings if necessary? No, sample_submission.csv showed 0/1 (int).
    # Wait, let's re-check sample_submission.csv
    # id,Heart Disease
    # 630000,0
    
    submission.to_csv('submission.csv', index=False)
    print("Submission saved to submission.csv")

if __name__ == "__main__":
    main()
