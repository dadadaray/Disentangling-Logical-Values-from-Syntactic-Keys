import sys, os
_PROJ_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _PROJ_ROOT not in sys.path:
    sys.path.insert(0, _PROJ_ROOT)

import os
import json
import random
import argparse
from pathlib import Path
from tqdm import tqdm
import glob

import torch
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer

from utils import nethook
from utils.layer_stats_mom2 import layer_stats

# =========================
# BLiMP CONFIG
# =========================
BLIMP_PARADIGMS = [
    "principle_A_c_command",
    "principle_A_domain_1",
    "principle_A_domain_2",
    "principle_A_domain_3",
    "principle_A_reconstruction",
    "principle_A_case_1",
    "principle_A_case_2",
    "regular_plural_subject_verb_agreement_1",
    "regular_plural_subject_verb_agreement_2",
    "irregular_plural_subject_verb_agreement_1",
    "irregular_plural_subject_verb_agreement_2",
]


def load_blimp_local(blimp_root, paradigms, max_samples_per_paradigm=300, seed=42):
    print(f"Loading BLiMP from: {blimp_root}")
    all_samples = []
    pattern = os.path.join(blimp_root, "*.jsonl")
    files = glob.glob(pattern)
    if not files:
        files = glob.glob(os.path.join(blimp_root, "*.json"))

    for file_path in files:
        file_name = Path(file_path).stem
        if paradigms and file_name not in paradigms:
            continue
        with open(file_path, "r", encoding="utf-8") as f:
            lines = f.readlines()
        if seed is not None:
            random.Random(seed).shuffle(lines)
        count = 0
        for line in lines:
            line = line.strip()
            if not line: continue
            try:
                data = json.loads(line)
                all_samples.append(data)
                count += 1
                if count >= max_samples_per_paradigm: break
            except:
                continue
    print(f"Loaded {len(all_samples)} samples in total.")
    return all_samples


def get_mom2_covariance(model, tok, layer_name, mom2_dataset, sample_n, stats_dir, stats_model_name):
    """
    Compute or load the MOM2 covariance matrix.
    """
    print(f"Loading MOM2 for: {layer_name}")
    print(f"Target file suffix expected: _{sample_n}.npz")

    # Force stats_model_name as the model_name parameter
    # so that layer_stats looks under stats_dir/stats_model_name/...
    stat = layer_stats(
        model,
        tok,
        layer_name,
        stats_dir=stats_dir,
        ds_name=mom2_dataset,
        to_collect=["mom2"],
        sample_size=sample_n,
        precision="float32",
        model_name=stats_model_name,  # Critical: must match the folder name
        download=False  # Do not attempt to download since the file exists locally
    )

    # stat.mom2.moment() returns the covariance matrix C
    C = stat.mom2.moment().float()
    return C


def extract_syntax_activations(model, tok, layer_name, sentences, device):
    """
    Extract activations at the same location where MOM2 is computed.
    For .mlp.down_proj, MOM2 uses the input (tr.input).
    """
    print(f"⚡ Extracting BLiMP activations from input of: {layer_name}...")
    acts = []
    model.eval()
    with torch.no_grad():
        for item in tqdm(sentences, desc="Extracting BLiMP"):
            text = item['sentence_good']
            inp = tok(text, return_tensors="pt").to(device)

            # Use retain_input=True to capture the input
            with nethook.Trace(model, layer_name, retain_input=True) as tr:
                _ = model(**inp)

                # tr.input is normally a tuple (tensor, )
                feat = tr.input
                if isinstance(feat, tuple):
                    feat = feat[0]  # [1, seq_len, dim]

                # Take the last token
                feat_mean = feat[0, -1, :].cpu()
                acts.append(feat_mean)

    return torch.stack(acts)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, required=True)
    parser.add_argument("--model_alias", type=str, default="model")
    parser.add_argument("--layer", type=int, required=True)
    parser.add_argument("--blimp_root", type=str, required=True)
    parser.add_argument("--stats_dir", type=str, required=True)
    parser.add_argument("--stats_model_name", type=str, required=True, help="Exact folder name in stats_dir")
    parser.add_argument("--output_dir", type=str, default="analysis_results")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # 1. Load Model
    print(f"Loading model: {args.model_name}...")
    model = AutoModelForCausalLM.from_pretrained(args.model_name, torch_dtype=torch.float16, device_map="auto")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    tokenizer.pad_token = tokenizer.eos_token

    # [Critical fix] Construct the full layer name to match the file naming convention
    # e.g. model.layers.6.mlp.down_proj
    layer_name = f"model.layers.{args.layer}.mlp.down_proj"

    # 2. Load MOM2 Covariance
    # sample_n must be 100000 here, otherwise the file name suffix won't match
    C = get_mom2_covariance(
        model,
        tokenizer,
        layer_name,
        mom2_dataset="wikipedia",
        sample_n=100000,
        stats_dir=args.stats_dir,
        stats_model_name=args.stats_model_name
    ).to(device)

    # 3. Eigen Decomposition
    print("Performing Eigen Decomposition...")
    S, V = torch.linalg.eigh(C)
    idx = torch.argsort(S, descending=True)
    S = S[idx]
    V = V[:, idx]

    # 4. Load BLiMP
    blimp_sentences = load_blimp_local(args.blimp_root, BLIMP_PARADIGMS, max_samples_per_paradigm=300)

    # 5. Extract Syntax Activations (At same site!)
    syntax_matrix = extract_syntax_activations(model, tokenizer, layer_name, blimp_sentences, device).to(device)

    # 6. Projection
    print("Projecting Syntax activations...")
    syntax_matrix = syntax_matrix.float()
    V = V.float()

    projections = []
    chunk_size = 200
    with torch.no_grad():
        for i in range(0, syntax_matrix.shape[0], chunk_size):
            chunk = syntax_matrix[i: i + chunk_size]
            proj = chunk @ V
            energy = proj ** 2
            projections.append(energy.cpu())

    all_energy = torch.cat(projections, dim=0)
    avg_grammar_energy = all_energy.mean(dim=0)

    # 7. Save
    os.makedirs(args.output_dir, exist_ok=True)
    save_path = os.path.join(
        args.output_dir,
        f"layer_mom2_{args.model_alias}_{args.layer}_spectral_data.pt"
    )

    torch.save({
        "layer": args.layer,
        "layer_name_full": layer_name,
        "model": args.model_name,
        "stats_folder": args.stats_model_name,
        "eigenvalues": S.cpu(),
        "grammar_projection": avg_grammar_energy,
        "num_samples": len(blimp_sentences)
    }, save_path)

    print(f"✅ Done. Saved to {save_path}")


if __name__ == "__main__":
    main()