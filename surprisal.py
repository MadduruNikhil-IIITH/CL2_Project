import torch
from transformers import GPT2TokenizerFast, GPT2LMHeadModel
from transformers import BertTokenizerFast, BertForMaskedLM
import numpy as np

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Surprisal device: {device}")

# GPT-2
gpt2_tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")
gpt2_tokenizer.pad_token = gpt2_tokenizer.eos_token
gpt2_model = GPT2LMHeadModel.from_pretrained("gpt2").to(device)

# BERT
bert_tokenizer = BertTokenizerFast.from_pretrained("bert-base-uncased")
bert_model = BertForMaskedLM.from_pretrained("bert-base-uncased").to(device)

gpt2_model.eval()
bert_model.eval()

@torch.no_grad()
def get_surprisal_features(sentence: str) -> dict:
    if not sentence.strip():
        return {
            "gpt2_surprisal_mean":0,"gpt2_surprisal_max":0,"gpt2_surprisal_min":0,
            "gpt2_surprisal_var":0,"gpt2_surprisal_sum":0, "gpt2_surprisal_std":0,
            "bert_surprisal_mean":0,"bert_surprisal_max":0,"bert_surprisal_min":0,
            "bert_surprisal_var":0,"bert_surprisal_sum":0,"bert_surprisal_std":0,
            # "gpt2_bigram_avg":5.0, "gpt2_trigram_avg":5.0
        }

    # === 1. TOKEN-LEVEL ===
    inputs = gpt2_tokenizer(sentence, return_tensors="pt", truncation=True, max_length=1024).to(device)
    outputs = gpt2_model(**inputs, labels=inputs["input_ids"])
    shift_logits = outputs.logits[..., :-1, :].contiguous()
    shift_labels = inputs["input_ids"][..., 1:].contiguous()
    loss_fct = torch.nn.CrossEntropyLoss(reduction='none', ignore_index=gpt2_tokenizer.pad_token_id)
    gpt_losses = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
    gpt_losses = gpt_losses.view(shift_labels.shape)
    mask = shift_labels != gpt2_tokenizer.pad_token_id
    gpt_losses = gpt_losses[mask].cpu().numpy()

    # BERT token-level
    inputs_b = bert_tokenizer(sentence, return_tensors="pt", truncation=True, max_length=510).to(device)
    seq_len = inputs_b["input_ids"].shape[1]
    bert_losses = []
    for i in range(1, seq_len - 1):
        masked = inputs_b["input_ids"].clone()
        masked[0, i] = bert_tokenizer.mask_token_id
        pred = bert_model(masked).logits[0, i]
        prob = torch.softmax(pred, dim=-1)[inputs_b["input_ids"][0, i]]
        bert_losses.append(-torch.log(prob + 1e-10).item())
    bert_losses = np.array(bert_losses) if bert_losses else np.array([0.0])

    # === 2. AVERAGE BI-GRAM & TRI-GRAM SURPRISAL (all positions) ===
    # tokens = gpt2_tokenizer.tokenize(sentence)

    # # GPT-2: average bi-gram/tri-gram loss
    # gpt2_bigram_losses = []
    # gpt2_trigram_losses = []
    # for i in range(len(tokens)):
    #     if i >= 1:
    #         inp = gpt2_tokenizer(" ".join(tokens[i-1:i+1]), return_tensors="pt").to(device)
    #         loss = gpt2_model(**inp, labels=inp["input_ids"]).loss.item()
    #         gpt2_bigram_losses.append(loss)
    #     if i >= 2:
    #         inp = gpt2_tokenizer(" ".join(tokens[i-2:i+1]), return_tensors="pt").to(device)
    #         loss = gpt2_model(**inp, labels=inp["input_ids"]).loss.item()
    #         gpt2_trigram_losses.append(loss)

    # gpt2_bigram_avg = np.mean(gpt2_bigram_losses) if gpt2_bigram_losses else 5.0
    # gpt2_trigram_avg = np.mean(gpt2_trigram_losses) if gpt2_trigram_losses else 5.0

    # === 3. STATS ===
    def stats(arr):
        arr = np.clip(arr, 0, 30)
        if len(arr) == 0: arr = np.array([0.0])
        return {
            "mean": float(np.mean(arr)),
            "max": float(np.max(arr)),
            "min": float(np.min(arr)),
            "var": float(np.var(arr)),
            "sum": float(np.sum(arr)),
            "std": float(np.std(arr))
        }

    g = stats(gpt_losses)
    b = stats(bert_losses)

    return {
        # Approved token-level
        "gpt2_surprisal_mean": g["mean"], "gpt2_surprisal_max": g["max"],
        "gpt2_surprisal_min": g["min"], "gpt2_surprisal_var": g["var"], 
        "gpt2_surprisal_sum": g["sum"], "gpt2_surprisal_std": g["std"],
        "bert_surprisal_mean": b["mean"], "bert_surprisal_max": b["max"],
        "bert_surprisal_min": b["min"], "bert_surprisal_var": b["var"], 
        "bert_surprisal_sum": b["sum"], "bert_surprisal_std": b["std"],
        
        # Brilliant extension
        # "gpt2_bigram_avg_surprisal": gpt2_bigram_avg,
        # "gpt2_trigram_avg_surprisal": gpt2_trigram_avg
    }