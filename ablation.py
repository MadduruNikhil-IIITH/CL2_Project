import json
import os
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
import pandas as pd

def run_ablation(run_folder):
    df = pd.read_csv(f"{run_folder}/sentences_with_features.csv")
    X = df.drop(columns=["para_id", "sent_id", "sentence", "label"])
    y = df["label"]

    # Load top features from full model
    top20 = pd.read_csv(f"{run_folder}/top20_features.csv")
    top10_features = top20["feature"].head(10).tolist()

    STATS_FILE = "results/run_statistics.json"
    
    if not os.path.exists(STATS_FILE):
        print("No results found. Run main.py first.")
        return

    with open(STATS_FILE, "r") as f:
        data = json.load(f)

    run_data_key = run_folder.split("_")[1]
    run_data = data["runs"].get(run_data_key, None)
    
    # Full model performance (from your run)
    full_acc = run_data['accuracy']
    full_f1 = run_data['f1_answer']

    # Top-10 only
    model = LogisticRegression(C=1.0, class_weight='balanced', max_iter=1000)
    X_cleaned = X[top10_features].fillna(0).replace([np.inf, -np.inf], 0)
    model.fit(X_cleaned, y)
    preds = model.predict(X_cleaned)
    top10_acc = accuracy_score(y, preds)
    top10_f1 = f1_score(y, preds)

    print("ABLATION STUDY - " + run_data_key)
    print(f"Full Model (all features):  Acc {full_acc:.4f} | F1 {full_f1:.4f}")
    print(f"Top-10 Features Only:       Acc {top10_acc:.4f} | F1 {top10_f1:.4f}")

    with open(f"{run_folder}/ablation_results.txt", "w") as f:
        f.write(f"Full: {full_acc:.4f}/{full_f1:.4f}\n")
        f.write(f"Top10: {top10_acc:.4f}/{top10_f1:.4f}\n")
        
# run_ablation("results/run_100_passages")
# print("-"*50)
# run_ablation("results/run_250_passages")
# print("-"*50)
# run_ablation("results/run_500_passages")
# print("-"*50)
# run_ablation("results/run_1000_passages")
# print("-"*50)
# run_ablation("results/run_2000_passages")