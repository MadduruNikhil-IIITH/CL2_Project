import nltk
import re
import numpy as np
from collections import Counter
from surprisal import get_surprisal_features

RE_CLEAN = re.compile(r"[^a-zA-Z\s]")
RE_SPACE = re.compile(r"\s+")

def clean_text(text: str) -> str:
    text = re.sub(r"http\S+", " ", text)
    text = re.sub(r"\d+", " ", text)
    text = RE_CLEAN.sub(" ", text)
    return RE_SPACE.sub(" ", text).strip().lower()

def extract_linguistic_features(sentence: str, position: int, total_sentences: int) -> dict:
    original = sentence.strip()
    cleaned = clean_text(original) or "empty"
    
    tokens = nltk.word_tokenize(cleaned)
    tagged = nltk.pos_tag(tokens, tagset="universal")
    
    words = [t for t in tokens if t.isalpha()]
    if not words:
        words = ["x"]
    
    word_lengths = [len(w) for w in words]
    types = set(words)
    pos_counts = Counter(p for _, p in tagged)
    total_pos = sum(pos_counts.values()) or 1
    total_tokens = len(tokens) or 1

    causal = {"because","since","as","so","therefore","thus","hence"}
    contrast = {"but","however","although","though","yet","still","nevertheless"}
    ne_count = sum(1 for w in original.split() if w.isalpha() and w[0].isupper() and len(w)>1)

    ling_features = {
        "avg_word_length": np.mean(word_lengths),
        "sentence_length_words": len(words),
        "sentence_position": position,
        "sentence_position_norm": position / total_sentences,
        "type_token_ratio": len(types) / len(words),
        "lexical_density": sum(1 for _,p in tagged if p in {"NOUN","VERB","ADJ","ADV"}) / len(words),
        "noun_ratio": pos_counts["NOUN"]/total_pos,
        "verb_ratio": pos_counts["VERB"]/total_pos,
        "adj_ratio": pos_counts["ADJ"]/total_pos,
        "pronoun_ratio": pos_counts["PRON"]/total_pos,
        "noun_verb_ratio": pos_counts["NOUN"]/(pos_counts["VERB"]+1e-8),
        "causal_marker_ratio": sum(w in causal for w in tokens)/total_tokens,
        "contrast_marker_ratio": sum(w in contrast for w in tokens)/total_tokens,
        "named_entity_density": ne_count / len(words),
    }
    
    # # === MORE SIMPLE FEATURES ===
    # # 1. Readability (Flesch Reading Ease — simple formula)
    # avg_sent_len = len(words)  # Proxy for sentence length
    # avg_word_len = np.mean(word_lengths)
    # flesch = 206.835 - 1.015 * avg_word_len - 84.6 * (1 / avg_sent_len) if avg_sent_len > 0 else 0
    # flesch = np.clip(flesch, 0, 100)  # Normalize

    # # 2. Punctuation density (simple count)
    # punct_count = len(re.findall(r'[.,!?;:]', original))
    # punctuation_density = punct_count / len(words) if len(words) > 0 else 0

    # # 3. Character diversity (MTLD proxy — simple TTR on chars)
    # char_types = set(original.lower())
    # char_ttr = len(char_types) / len(original) if original else 0

    # # 4. Extended POS (adverb/noun ratio — from pos_features.py in repo)
    # adverb_ratio = pos_counts["ADV"] / (pos_counts["NOUN"] + 1e-8)

    # genre_features = {
    #     # "flesch_reading_ease": flesch,
    #     # "punctuation_density": punctuation_density,
    #     # "char_diversity_ttr": char_ttr,
    #     # "adverb_noun_ratio": adverb_ratio,
    # }

    surp = get_surprisal_features(original)

    # features = {**ling_features, **surp, **genre_features}
    features = {**ling_features, **surp}
    
    # Safety
    for k in features:
        if isinstance(features[k], float) and (np.isnan(features[k]) or np.isinf(features[k])):
            features[k] = 0.0

    return features