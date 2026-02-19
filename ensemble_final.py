import pandas as pd
import numpy as np
import glob
import os

def main():
    # Identify all probability files
    prob_files = glob.glob('test_probs_*.csv')
    
    if not prob_files:
        print("No probability files found. Run advanced_models.py and advanced_pytabkit.py first.")
        return

    print(f"Ensembling {len(prob_files)} models...")
    for f in prob_files:
        print(f"  - {f}")

    # Load all probabilities
    probs = []
    ids = None
    
    # We can assign custom weights if needed, for now let's do a weighted average 
    # based on typical performance of models in such tasks.
    # Often, CatBoost and LightGBM are very strong.
    
    for f in prob_files:
        df = pd.read_csv(f)
        if ids is None:
            ids = df['id']
        probs.append(df['prob'].values)

    # Simple average
    avg_prob = np.mean(probs, axis=0)
    
    # Majority vote on binary predictions
    binary_preds = []
    for p in probs:
        binary_preds.append((p > 0.5).astype(int))
    
    majority_vote = (np.sum(binary_preds, axis=0) > (len(prob_files) / 2)).astype(int)

    # Save ensemble based on average probabilities
    submission_avg = pd.DataFrame({
        'id': ids,
        'Heart Disease': (avg_prob > 0.5).astype(int)
    })
    submission_avg.to_csv('submission_final_ensemble_avg.csv', index=False)
    print("Ensemble submission (Average) saved to submission_final_ensemble_avg.csv")

    # Save ensemble based on majority vote
    submission_vote = pd.DataFrame({
        'id': ids,
        'Heart Disease': majority_vote
    })
    submission_vote.to_csv('submission_final_ensemble_vote.csv', index=False)
    print("Ensemble submission (Majority Vote) saved to submission_final_ensemble_vote.csv")

if __name__ == "__main__":
    main()
