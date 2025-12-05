import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import joblib

with open("results/run_statistics.json", "r") as f:
    stats = json.load(f)

runs = stats["runs"]
passages = sorted([int(k) for k in runs.keys()])

# === 1. Scaling Subplots ===
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 10))

# Top: Accuracy, Precision, Recall (%)
acc = [runs[str(p)]["accuracy"]*100 for p in passages]
prec = [runs[str(p)]["precision"]*100 for p in passages]
rec = [runs[str(p)]["recall"]*100 for p in passages]

ax1.plot(passages, acc, 'o-', label="Accuracy", markersize=8)
ax1.plot(passages, prec, 's-', label="Precision", markersize=8)
ax1.plot(passages, rec, '^-', label="Recall", markersize=8)
ax1.set_ylabel("Percentage (%)")
ax1.set_title("Performance Scaling: Accuracy, Precision, Recall")
ax1.legend()
ax1.grid(True, alpha=0.3)

# Bottom: F1-score
f1 = [runs[str(p)]["f1_answer"] for p in passages]
ax2.plot(passages, f1, 'd-', color='purple', markersize=8, linewidth=2.5)
ax2.set_ylabel("F1-score")
ax2.set_xlabel("Number of Passages")
ax2.set_title("Performance Scaling: F1-score (Answer Class)")
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig("results/scaling_subplot.png", dpi=300, bbox_inches='tight')
plt.close()

def generate_additional_plots(run_folder):
    top20 = pd.read_csv(f"{run_folder}/top20_features.csv")

    plt.figure(figsize=(10,8))
    sns.barplot(x="coefficient", y="feature", data=top20)
    plt.title("Top 20 Features by Coefficient Magnitude")
    plt.xlabel("Coefficient")
    plt.ylabel("Feature")
    plt.tight_layout()
    plt.savefig(f"{run_folder}/feature_importance.png")
    plt.close()

    # 3. Confusion Matrix and Prediction Histogram (load data and model)
    df = pd.read_csv(f"{run_folder}/sentences_with_features.csv")
    cols = [c for c in df.columns if c not in ["para_id","sent_id","sentence","label"]]
    X = df[cols].fillna(0).replace([np.inf, -np.inf], 0)
    y = df["label"]

    model = joblib.load(f"{run_folder}/model.pkl")
    scaler = joblib.load(f"{run_folder}/scaler.pkl")

    X_s = scaler.transform(X)
    preds = model.predict(X_s)
    probs = model.predict_proba(X_s)[:, 1]  # Probability for class 1

    # Confusion Matrix
    cm = confusion_matrix(y, preds)
    disp = ConfusionMatrixDisplay(cm, display_labels=["Non-Answer", "Answer"])
    disp.plot(cmap=plt.cm.Blues)
    plt.title("Confusion Matrix - Full Model")
    plt.savefig(f"{run_folder}/confusion_matrix.png")
    plt.close()

    # Prediction Probability Histogram
    plt.figure(figsize=(8,6))
    sns.histplot(probs[y==0], bins=20, kde=True, color="blue", label="Non-Answer")
    sns.histplot(probs[y==1], bins=20, kde=True, color="green", label="Answer")
    plt.title("Prediction Probability Distribution")
    plt.xlabel("Predicted Probability (Answer Class)")
    plt.ylabel("Frequency")
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{run_folder}/prediction_histogram.png")
    plt.close()

    print("All plots generated successfully in the run folder.")
    
generate_additional_plots("results/run_100_passages")
generate_additional_plots("results/run_250_passages")
generate_additional_plots("results/run_500_passages")
generate_additional_plots("results/run_1000_passages")
generate_additional_plots("results/run_2000_passages")