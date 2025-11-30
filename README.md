# CL2 Project

This repository contains code to analyze passages from the SQuAD dataset and build sentence-level features (linguistic + surprisal) to classify whether a sentence contains the answer.

Overview
- The pipeline loads SQuAD-formatted JSON (data/train.json), extracts sentences, computes surprisal features using GPT-2/BERT (using HuggingFace Transformers), extracts hand-crafted linguistic features, trains a classifier (logistic regression), and runs ablation and PCA analyses.
- The project supports GPU acceleration for surprisal calculation if a CUDA-capable GPU is available.

Folder structure
- `ablation.py` — Run ablation experiments using top features and save metrics to `results`.
- `classifier.py` — Train a logistic regression model on features and save `model.pkl`, `scaler.pkl`, and `top20_features.csv`.
- `data_stats_and_sampling.py` — Load SQuAD `train.json`, generate dataset visualizations and simple dataset stats.
- `feature_extractor.py` — Cleanup text and compute linguistic features used by the classifier.
- `main.py` — Orchestrates the full pipeline (load, feature extraction, surprisal, training, results & graphs).
- `pca.py` — Perform PCA analysis on feature space and save visualizations and analysis to `results`.
- `surprisal.py` — Compute token-level surprisal using GPT-2 and BERT. Uses transformers and PyTorch; can use GPU if available.
- `data/` — Place the `train.json` and `dev.json` (SQuAD) files here.
- `results/` — Output directory with `run_<N>_passages` subfolders (contains raw sentences, features, plots, model files, etc.).
- `menv/` — (Optional) local Python virtual environment included in the workspace.

Getting started

1. Create a virtual environment (optional, recommended):

```powershell
python -m venv menv
./menv/Scripts/Activate.ps1
```

2. Install dependencies: copy the dependencies below into a `requirements.txt` or install manually:

```powershell
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118  # or pick the right CUDA version from pytorch.org
pip install transformers nltk pandas scikit-learn matplotlib seaborn joblib tqdm
```

Tip: visit https://pytorch.org/ to select the correct wheel for your CUDA version (e.g., `cu118`, `cu121`, or `cpu`).

3. Download the SQuAD dataset files into the `data/` directory (`train.json` and `dev.json`).

4. Run the full pipeline:

```powershell
python main.py
```

Notes
- `main.py` has a `MAX_PARAGRAPHS` variable near the top — adjust this to control how many SQuAD passages are used for a run (default is 2000). The script will create a `results/run_<N>_passages` subfolder with outputs.
- Surprisal computation uses Transformers and can be slow on CPU. If you have a CUDA GPU, PyTorch will use it automatically; otherwise CPU will be used.
- For reproducible results, consider creating a `requirements.txt` file with pinned versions.


