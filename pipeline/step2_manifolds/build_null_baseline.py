import sys, os
_PROJ_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _PROJ_ROOT not in sys.path:
    sys.path.insert(0, _PROJ_ROOT)

import torch
import os
import random
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer
from utils import nethook


MODEL_NAME = "/data/users/yanrongen/AnyEdit/LLM-Llama-3-8B-Instruct"
LAYERS = [12, 13, 14, 15, 16]
SAVE_DIR = "rema_matrices"
NUM_SAMPLES = 100

RANDOM_WORDS = [
    "apple", "run", "sky", "blue", "philosophy", "table", "entropy", "go",
    "is", "the", "a", "of", "random", "noise", "matrix", "tensor", "cat",
    "fly", "eat", "logic", "zero", "dimension", "space", "time", "banana"
]


def get_random_nonsense():
    length = random.randint(10, 30)
    words = [random.choice(RANDOM_WORDS) for _ in range(length)]
    return " ".join(words) + "."


def main():
    print(f"Generating Nonsense Manifold (The Strict Baseline)...")

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME, device_map="auto", torch_dtype=torch.float32, low_cpu_mem_usage=True
    ).eval()
    tok = AutoTokenizer.from_pretrained(MODEL_NAME, padding_side='left')
    tok.pad_token = tok.eos_token

    prompts = [get_random_nonsense() for _ in range(NUM_SAMPLES)]
    print(f"Example Nonsense: {prompts[0]}")

    collected_vecs = {l: [] for l in LAYERS}

    batch_size = 8
    for i in range(0, len(prompts), batch_size):
        batch = prompts[i:i + batch_size]
        inputs = tok(batch, return_tensors="pt", padding=True).to(model.device)

        with torch.no_grad():
            with nethook.TraceDict(model, [f"model.layers.{l}" for l in LAYERS]) as tr:
                _ = model(**inputs)

            for l in LAYERS:
                out = tr[f"model.layers.{l}"].output
                if isinstance(out, tuple): out = out[0]

                last_token_idxs = inputs.attention_mask.sum(dim=1) - 1
                feats = out[torch.arange(len(batch)), last_token_idxs, :]
                collected_vecs[l].append(feats.cpu())

    if not os.path.exists(SAVE_DIR): os.makedirs(SAVE_DIR)

    for l in LAYERS:
        if not collected_vecs[l]: continue
        X = torch.cat(collected_vecs[l], dim=0).float()  # [N, Dim]

        # SVD
        # Center the data
        X_centered = X - X.mean(dim=0)
        k = min(4096, X.shape[0])
        _, _, V = torch.svd_lowrank(X_centered, q=k)

        save_path = f"{SAVE_DIR}/rema_U_nonsense_L{l}.pt"
        torch.save(V, save_path)
        print(f"✅ Saved Nonsense Manifold: {save_path}")


if __name__ == "__main__":
    main()