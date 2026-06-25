import sys, os
_PROJ_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _PROJ_ROOT not in sys.path:
    sys.path.insert(0, _PROJ_ROOT)

import torch
import os
import json
import numpy as np
import argparse
import glob
from transformers import AutoModelForCausalLM, AutoTokenizer
from utils import nethook


def parse_args():
    parser = argparse.ArgumentParser(description="Generic Curvature Analysis Tool")
    parser.add_argument("--model_path", type=str, default="/data/users/yanrongen/AnyEdit/LLM-Llama-3-8B-Instruct")
    parser.add_argument("--matrix_dir", type=str, required=True, help="Directory containing upd_raw/upd_trim .pt files")
    parser.add_argument("--dataset_path", type=str, required=True, help="Path to the probe dataset (.json)")
    parser.add_argument("--dataset_name", type=str, default="custom",
                        help="Name for logging (e.g., gsm8k, math, mquake)")
    parser.add_argument("--layers", type=int, nargs="+", default=[12, 13, 14, 15, 16])
    parser.add_argument("--batch_size", type=int, default=10, help="Number of probe samples to use")
    return parser.parse_args()


def load_dataset_dynamic(path, tokenizer, batch_size=10):

    print(f"Loading probe data from: {path}")
    if not os.path.exists(path):
        raise FileNotFoundError(f"Dataset not found: {path}")

    questions = []

    with open(path, 'r') as f:
        try:
            data = json.load(f)
        except json.JSONDecodeError:
            # Try loading as jsonl
            f.seek(0)
            data = [json.loads(line) for line in f]


    for item in data:
        if len(questions) >= batch_size: break

        q = None
        # 1. MATH format
        if 'problem' in item:
            q = item['problem']
        # 2. GSM8K format
        elif 'question' in item:
            q = item['question']
        # 3. MQuAKE format
        elif 'questions' in item and isinstance(item['questions'], list):
            q = item['questions'][0]
        elif 'requested_rewrite' in item:
            rr = item['requested_rewrite']
            if 'prompt' in rr and 'subject' in rr:
                q = rr['prompt'].format(rr['subject'])

        if q:
            questions.append(q)

    print(f"Loaded {len(questions)} probe samples.")


    prompts = [
        f"<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\n{q}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
        for q in questions
    ]

    inputs = tokenizer(prompts, return_tensors="pt", padding=True, truncation=True).to("cuda")
    return inputs


def compute_loss(model, inputs):
    with torch.no_grad():
        outputs = model(**inputs, labels=inputs["input_ids"])
        return outputs.loss.item()


def measure_sensitivity(model, layer_idx, update_matrix, inputs, epsilon=1e-2):
    layer_name = f"model.layers.{layer_idx}.mlp.down_proj"
    weights = nethook.get_parameter(model, f"{layer_name}.weight")

    d = update_matrix.to(weights.device).float()

    d_norm = d / (d.norm() + 1e-8)

    scale = weights.norm().item() * epsilon
    perturbation = d_norm * scale

    # 1. Base Loss
    l0 = compute_loss(model, inputs)

    # 2. Perturb +
    weights.data += perturbation
    l_plus = compute_loss(model, inputs)

    # 3. Restore
    weights.data -= perturbation

    return l_plus - l0



def reconstruct_matrix(pt_path, device="cuda"):
    data = torch.load(pt_path, map_location=device)
    u = data["u"].float()
    s = data["s"].float()
    v = data["v"].float()

    # Reconstruct: U @ diag(S) @ V.T
    # [Out, 1] @ [1] @ [1, In] -> [Out, In]
    matrix = (u * s) @ v.T
    return matrix


def main():
    args = parse_args()

    print(f"Loading model: {args.model_path}")
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(args.model_path, device_map="auto")

    inputs = load_dataset_dynamic(args.dataset_path, tokenizer, batch_size=args.batch_size)

    print(f"\n⚔️  Curvature Analysis: {args.dataset_name.upper()} Landscape ⚔️")
    print(f"Scanning vectors in directory: {args.matrix_dir}")


    print(f"{'Layer':<6} | {'Raw Sens.':<12} | {'TRIM Sens.':<12} | {'Reduction':<10}")
    print("-" * 50)

    total_raw = 0
    total_trim = 0

    for layer in args.layers:

        search_pattern = os.path.join(args.matrix_dir, f"factors_raw_L{layer}_sample*.pt")
        raw_files = glob.glob(search_pattern)

        if not raw_files:
            print(f"L{layer:<5} | No data found")
            continue

        selected_files = raw_files[:20]
        n_count = len(selected_files)

        avg_raw = 0
        avg_trim = 0

        for f_raw in selected_files:
            f_trim = f_raw.replace("factors_raw", "factors_trim")
            if not os.path.exists(f_trim): continue


            upd_raw = reconstruct_matrix(f_raw, device=model.device)
            upd_trim = reconstruct_matrix(f_trim, device=model.device)

            delta_raw = measure_sensitivity(model, layer, upd_raw, inputs)
            delta_trim = measure_sensitivity(model, layer, upd_trim, inputs)

            avg_raw += delta_raw
            avg_trim += delta_trim

        if n_count > 0:
            avg_raw /= n_count
            avg_trim /= n_count

            ratio = avg_raw / (avg_trim + 1e-8)
            print(f"L{layer:<5} | {avg_raw:.4e}   | {avg_trim:.4e}   | {ratio:.1f}x")

            total_raw += avg_raw
            total_trim += avg_trim

    print("-" * 50)
    if total_trim > 0:
        print(f"Overall Improvement: {total_raw / total_trim:.2f}x flatter")
    else:
        print("No valid data processed.")


if __name__ == "__main__":
    main()