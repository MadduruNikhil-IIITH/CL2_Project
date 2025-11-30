import json
import random
import nltk
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import os

SEED = 2024901010
random.seed(SEED)

def load_and_visualize(max_paragraphs: int | None = 100) -> tuple[list[dict], int]:
    """
    Loads SQuAD, generates 4 beautiful graphs,
    Saves them in results/run_XXX_passages/dataset_visualizations.png
    Returns: (paragraphs, passage_count)
    """
    print("Loading train.json + generating visualizations...")
    with open("data/train.json") as f:
        data = json.load(f)["data"]

    # Count TOTAL available passages
    total_available = sum(len(article["paragraphs"]) for article in data)
    print(f"Total available passages in SQuAD : {total_available:,}")

    sent_lens, para_lens, positions, labels = [], [], [], []
    paragraphs = []
    pid = 0

    for article in data:
        for para in article["paragraphs"]:
            ctx = para["context"]
            sents = nltk.sent_tokenize(ctx)
            para_lens.append(len(sents))

            spans = [(a["answer_start"], a["answer_start"] + len(a["text"]))
                     for qa in para["qas"] for a in qa["answers"]]

            pos = 0
            for i, s in enumerate(sents):
                end = pos + len(s)
                label = any(not (end <= st or pos >= en) for st, en in spans)
                sent_lens.append(len(nltk.word_tokenize(s)))
                positions.append(i + 1)
                labels.append(label)
                pos += len(s) + 1

            if max_paragraphs is None or pid < max_paragraphs:
                paragraphs.append(para)
            pid += 1
            if max_paragraphs and pid >= max_paragraphs:
                break
        if max_paragraphs and pid >= max_paragraphs:
            break

    passage_count = len(paragraphs)
    run_folder = f"results/run_{passage_count}_passages"
    os.makedirs(run_folder, exist_ok=True)

    # === 4 GRAPHS ===
    plt.style.use('seaborn-v0_8')
    fig, ax = plt.subplots(2, 2, figsize=(14, 10))

    ax[0,0].hist(sent_lens, bins=50, color='skyblue', alpha=0.8, edgecolor='black')
    ax[0,0].set_title("Sentence Length Distribution (words)")
    ax[0,0].set_xlabel("Number of words")
    ax[0,0].set_ylabel("Frequency")

    ax[0,1].hist(para_lens, bins=40, color='lightgreen', alpha=0.8, edgecolor='black')
    ax[0,1].set_title("Paragraph Length Distribution (sentences)")
    ax[0,1].set_xlabel("Number of sentences")

    rate = pd.Series(labels, index=positions).groupby(level=0).mean().head(20)
    ax[1,0].plot(rate.index, rate.values, 'o-', color='coral', linewidth=2, markersize=6)
    ax[1,0].set_title("Answer Probability by Sentence Position")
    ax[1,0].set_xlabel("Position in Paragraph")
    ax[1,0].set_ylabel("P(Contains Answer)")

    sns.boxplot(x=labels, y=sent_lens, ax=ax[1,1], palette="Set2")
    ax[1,1].set_title("Sentence Length: Answer vs Non-Answer")
    ax[1,1].set_xlabel("Contains Answer?")
    ax[1,1].set_xticklabels(["No", "Yes"])

    plt.tight_layout()
    viz_path = f"{run_folder}/dataset_visualizations.png"
    plt.savefig(viz_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Graphs saved: {viz_path}")
    
    # Stats
    stats = {
        "used_passages": passage_count,
        "total_sentences": len(sent_lens),
        "answer_sentences": sum(labels),
        "answer_ratio": np.mean(labels),
        "avg_sentence_length": np.mean(sent_lens),
        "avg_paragraph_length": np.mean(para_lens)
    }

    print(f"Using {passage_count:,} / {total_available:,} passages ({passage_count/total_available:.1%})")
    return paragraphs, passage_count, total_available, stats