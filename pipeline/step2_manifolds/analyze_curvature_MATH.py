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
import gc


def parse_args():
    parser = argparse.ArgumentParser(description="Generic Curvature Analysis Tool")
    parser.add_argument("--model_path", type=str, default="/data/users/yanrongen/AnyEdit/LLM-Llama-3-8B-Instruct")
    parser.add_argument("--matrix_dir", type=str, required=True, help="Directory containing factors_raw/trim .pt files")
    parser.add_argument("--dataset_path", type=str, required=True, help="Path to the probe dataset (.json)")
    parser.add_argument("--dataset_name", type=str, default="custom", help="Name for logging")
    parser.add_argument("--layers", type=int, nargs="+", default=[12, 13, 14, 15, 16])

    # [Key change] Separate out Batch Size
    parser.add_argument("--batch_size", type=int, default=1, help="Forward pass batch size (keep small for GPU)")
    parser.add_argument("--probe_limit", type=int, default=50, help="Total number of probe questions to test")

    return parser.parse_args()


def load_dataset_list(path, limit=50):
    """Load data and return a list of individual questions."""
    print(f"Loading probe data from: {path}")
    if not os.path.exists(path):
        raise FileNotFoundError(f"Dataset not found: {path}")

    questions = []
    with open(path, 'r') as f:
        try:
            data = json.load(f)
        except json.JSONDecodeError:
            f.seek(0)
            data = [json.loads(line) for line in f]

    for item in data:
        if len(questions) >= limit: break
        q = None
        if 'problem' in item:
            q = item['problem']
        elif 'question' in item:
            q = item['question']
        elif 'questions' in item:
            q = item['questions'][0]
        elif 'requested_rewrite' in item:
            rr = item['requested_rewrite']
            q = rr['prompt'].format(rr['subject'])
        if q: questions.append(q)

    print(f"Loaded {len(questions)} total probe samples.")
    return questions


def compute_loss(model, inputs):
    with torch.no_grad():
        outputs = model(**inputs, labels=inputs["input_ids"])
        return outputs.loss.item()


def measure_sensitivity(model, layer_idx, update_matrix, tokenizer, questions):
    """
    Compute average Loss sensitivity over a set of questions.
    """
    layer_name = f"model.layers.{layer_idx}.mlp.down_proj"
    weights = nethook.get_parameter(model, f"{layer_name}.weight")

    # Prepare perturbation vector
    d = update_matrix.to(weights.device).to(weights.dtype)
    d_norm = d / (d.norm() + 1e-8)
    scale = weights.norm().item() * 0.01  # epsilon = 1%
    perturbation = d_norm * scale

    total_delta = 0
    count = 0

    # [Key] Test one question at a time to prevent OOM
    # questions here is a list of strings
    for q in questions:
        # Construct a single input
        prompts = [
            f"<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\n{q}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"]
        inputs = tokenizer(prompts, return_tensors="pt", padding=True, truncation=True, return_attention_mask=True).to(
            "cuda")

        # 1. Base Loss
        l0 = compute_loss(model, inputs)

        # 2. Perturb +
        weights.data += perturbation
        l_plus = compute_loss(model, inputs)

        # 3. Restore
        weights.data -= perturbation

        # [Key] Take the absolute value! We need to know whether it "moved"
        total_delta += abs(l_plus - l0)
        count += 1

    del d, perturbation
    return total_delta / (count + 1e-8)


def reconstruct_matrix(pt_path, device="cuda"):
    data = torch.load(pt_path, map_location=device)
    u = data["u"].to(dtype=torch.float16)
    s = data["s"].to(dtype=torch.float16)
    v = data["v"].to(dtype=torch.float16)
    return (u * s) @ v.T


def main():
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
    args = parse_args()

    print(f"Loading model: {args.model_path}")
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(args.model_path, device_map="auto", torch_dtype=torch.float16)

    # 1. Load 50 questions
    questions = load_dataset_list(args.dataset_path, limit=args.probe_limit)

    print(f"\n⚔️  Curvature Analysis: {args.dataset_name.upper()} Landscape ⚔️")
    print(f"Scanning vectors in: {args.matrix_dir}")
    print(f"{'Layer':<6} | {'Raw Sens.':<12} | {'TRIM Sens.':<12} | {'Reduction':<10}")
    print("-" * 50)

    total_ratio = 0
    valid_layers = 0

    for layer in args.layers:
        search_pattern = os.path.join(args.matrix_dir, f"factors_raw_L{layer}_sample*.pt")
        raw_files = glob.glob(search_pattern)

        if not raw_files:
            print(f"L{layer:<5} | No data found")
            continue

        # Only take the first 5 vectors for testing (each vector tests 50 questions, sufficient computation)
        selected_files = raw_files[:5]

        layer_raw_sens = 0
        layer_trim_sens = 0

        for f_raw in selected_files:
            f_trim = f_raw.replace("factors_raw", "factors_trim")
            if not os.path.exists(f_trim): continue

            try:
                # Test Raw
                upd_raw = reconstruct_matrix(f_raw, device=model.device)
                layer_raw_sens += measure_sensitivity(model, layer, upd_raw, tokenizer, questions)
                del upd_raw

                # Test Trim
                upd_trim = reconstruct_matrix(f_trim, device=model.device)
                layer_trim_sens += measure_sensitivity(model, layer, upd_trim, tokenizer, questions)
                del upd_trim

            except RuntimeError as e:
                if "out of memory" in str(e):
                    torch.cuda.empty_cache()
                    continue
                raise e

            torch.cuda.empty_cache()

        # Take average
        layer_raw_sens /= len(selected_files)
        layer_trim_sens /= len(selected_files)

        ratio = layer_raw_sens / (layer_trim_sens + 1e-9)
        print(f"L{layer:<5} | {layer_raw_sens:.4e}   | {layer_trim_sens:.4e}   | {ratio:.1f}x")

        total_ratio += ratio
        valid_layers += 1

    print("-" * 50)
    if valid_layers > 0:
        print(f"Average Reduction: {total_ratio / valid_layers:.1f}x")
    else:
        print("No valid data.")


if __name__ == "__main__":
    main()