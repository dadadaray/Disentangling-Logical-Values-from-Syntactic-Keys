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
import copy


def parse_args():
    parser = argparse.ArgumentParser(description="MQuAKE Multi-hop Reasoning Sensitivity Test")
    parser.add_argument("--model_path", type=str, default="/data/users/yanrongen/AnyEdit/LLM-Llama-3-8B-Instruct")
    # Note: This expects a folder path containing .pt files
    parser.add_argument("--matrix_dir", type=str, required=True, help="Directory containing REMA/Trim vectors")
    # This expects the path to mquake.json
    parser.add_argument("--dataset_path", type=str, required=True, help="Path to mquake.json")
    parser.add_argument("--layers", type=int, nargs="+", default=[12, 13, 14, 15, 16])
    parser.add_argument("--probe_limit", type=int, default=100, help="Number of multi-hop cases to test")
    return parser.parse_args()


def load_mquake_data(path, limit=50):
    """
    Load MQuAKE-format data: (Multi-hop Question, Counterfactual Answer).
    """
    print(f"Loading MQuAKE data from: {path}")
    if not os.path.exists(path):
        raise FileNotFoundError(f"Dataset not found: {path}")

    qa_pairs = []
    with open(path, 'r') as f:
        # MQuAKE is typically a large List
        try:
            data = json.load(f)
        except json.JSONDecodeError:
            # Compatible with jsonl format
            f.seek(0)
            data = [json.loads(line) for line in f]

    for item in data:
        if len(qa_pairs) >= limit: break

        # 1. Get Multi-hop Questions
        # MQuAKE's questions field is a list containing different phrasings of the same question
        # We just take the first one
        q_list = item.get('questions', [])
        if not q_list: continue
        question = q_list[0]

        # 2. Get the Multi-hop reasoning target answer (New Answer / Counterfactual Answer)
        # We want to test: does the REMA signal trigger the model's “counterfactual reasoning” circuit?
        # If no new_answer (non-edited sample), fall back to answer
        target_answer = item.get('new_answer', item.get('answer'))

        if question and target_answer:
            qa_pairs.append((question, target_answer))

    print(f"Loaded {len(qa_pairs)} MQuAKE reasoning pairs.")
    # Print one sample to confirm correct loading
    if qa_pairs:
        print(f"Sample Q: {qa_pairs[0][0]}")
        print(f"Sample A: {qa_pairs[0][1]}")

    return qa_pairs


def compute_masked_loss(model, tokenizer, question, answer):
    """
    Construct the prompt and compute loss over the Answer portion.
    Prompt: <User> Question <Assistant> Answer
    """
    # Llama-3 template
    prompt_template = "<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\n{}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
    prompt = prompt_template.format(question)
    full_text = prompt + answer

    # Encode
    enc_prompt = tokenizer(prompt, return_tensors="pt", add_special_tokens=False)
    enc_full = tokenizer(full_text, return_tensors="pt", add_special_tokens=False)

    input_ids = enc_full.input_ids.to(model.device)
    labels = input_ids.clone()

    # Mask the Prompt portion (set to -100)
    # Note: compute the prompt length
    prompt_len = enc_prompt.input_ids.shape[1]
    labels[:, :prompt_len] = -100

    with torch.no_grad():
        outputs = model(input_ids=input_ids, labels=labels)
        return outputs.loss.item()


def measure_ablation_sensitivity(model, layer_idx, update_matrix, tokenizer, qa_pairs):
    """
    Comparison test: REMA signal vs. random equal-energy noise.
    + [New] Real-time generation probe: observe actual outputs of the first few samples.
    """
    layer_name = f"model.layers.{layer_idx}.mlp.down_proj"
    weights = nethook.get_parameter(model, f"{layer_name}.weight")

    # --- 1. Prepare REMA vector (Signal) ---
    d_signal = update_matrix.to(weights.device).to(weights.dtype)
    signal_norm = d_signal.norm().item()
    if signal_norm == 0: return 0, 0

    # --- 2. Prepare Random vector (Noise) ---
    # Key: keep energy (Norm) identical, randomize direction
    torch.manual_seed(42 + layer_idx)  # Different layers use different random seeds
    d_noise = torch.randn_like(d_signal)
    d_noise = d_noise / (d_noise.norm() + 1e-8) * signal_norm

    # --- 3. Set perturbation strength (epsilon) ---
    # Perturb 1% of the weight Norm
    epsilon = weights.norm().item() * 0.01

    # Normalize perturbation vector
    pert_signal = (d_signal / (signal_norm + 1e-8)) * epsilon
    pert_noise = (d_noise / (d_noise.norm() + 1e-8)) * epsilon

    total_delta_signal = 0
    total_delta_noise = 0
    count = 0

    # Prompt template (for formatting during generation)
    prompt_tpl = "<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\n{}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"

    print(f"  > Probing {len(qa_pairs)} MQuAKE samples...")

    for q, a in qa_pairs:
        try:
            # 0. Base Loss
            l0 = compute_masked_loss(model, tokenizer, q, a)

            # =========================================================
            # A. Signal Sensitivity (REMA)
            # =========================================================
            weights.data += pert_signal  # <--- Inject REMA signal

            # Compute Loss
            l_signal = compute_masked_loss(model, tokenizer, q, a)

            # >>> [Insertion Point] Case Study generation test (first 5 samples only) <<<
            if count < 5:
                print(f"\n🔍 [Case Study L{layer_idx} | Signal] Q: {q}")
                # Construct a standard chat input
                prompt = prompt_tpl.format(q)
                inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

                # Generate (Greedy Search, Max 30 tokens)
                with torch.no_grad():
                    gen_ids = model.generate(
                        **inputs,
                        max_new_tokens=30,
                        do_sample=False,
                        pad_token_id=tokenizer.eos_token_id
                    )

                # Decode (only extract the newly generated portion)
                input_len = inputs.input_ids.shape[1]
                gen_text = tokenizer.decode(gen_ids[0][input_len:], skip_special_tokens=True)

                # [Fix]: Pre-process the string to avoid backslashes in f-string
                clean_output = gen_text.strip().replace('\n', ' ')

                print(f"   [Model Output]: {clean_output}")
                print(f"   [Target Answer]: {a}")

            weights.data -= pert_signal  # <--- Restore weights

            # =========================================================
            # B. Noise Sensitivity (Random)
            # =========================================================
            weights.data += pert_noise
            l_noise = compute_masked_loss(model, tokenizer, q, a)
            weights.data -= pert_noise  # Restore

            # Accumulate absolute changes
            total_delta_signal += abs(l_signal - l0)
            total_delta_noise += abs(l_noise - l0)
            count += 1

        except Exception as e:
            print(f"Skipping sample due to error: {e}")
            continue

    avg_sens_signal = total_delta_signal / (count + 1e-9)
    avg_sens_noise = total_delta_noise / (count + 1e-9)

    return avg_sens_signal, avg_sens_noise


def reconstruct_matrix(pt_path, device="cuda"):
    """Load and reconstruct matrix, compatible with various storage formats."""
    data = torch.load(pt_path, map_location=device)

    # Case A: SVD format (U, S, V)
    if "u" in data and "s" in data and "v" in data:
        u = data["u"].to(dtype=torch.float16)
        s = data["s"].to(dtype=torch.float16)
        v = data["v"].to(dtype=torch.float16)
        # REMA could be a projection matrix or an update; here we assume it's the Update Delta
        return (u * s) @ v.T

    # Case B: Directly stored matrix
    elif "update_matrix" in data:
        return data["update_matrix"].to(dtype=torch.float16)

    # Case C: Only V (Projection Matrix)
    # If you stored a projection matrix P, you need to decide how to convert it to a perturbation
    # Here we assume the test target is the Update Matrix
    # If the file has no Update Matrix, you may need to specify the logic manually
    else:
        # Try returning the first value that appears to be a Tensor
        for k, v in data.items():
            if isinstance(v, torch.Tensor):
                return v.to(dtype=torch.float16)
    raise ValueError(f"Unknown matrix format in {pt_path}")


def main():
    args = parse_args()

    # Load model
    print(f"Loading Model: {args.model_path}")
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        device_map="auto",
        torch_dtype=torch.float16
    )

    # Load MQuAKE data
    qa_pairs = load_mquake_data(args.dataset_path, limit=args.probe_limit)

    print("\n" + "=" * 80)
    print(f"⚔️  REMA Logic Verification: Multi-hop Reasoning Sensitivity  ⚔️")
    print(f"Target: Does REMA affect the 'Counterfactual Answer' more than Random Noise?")
    print("=" * 80)
    print(f"{'Layer':<6} | {'Signal Sens.':<12} | {'Noise Sens.':<12} | {'Ratio (S/N)':<15} | {'Verdict'}")
    print("-" * 75)

    for layer in args.layers:
        # Find matching matrix file (adjust the wildcard pattern based on your naming convention)
        # Example filenames: factors_trim_L12.pt or rema_update_L12.pt
        pattern = os.path.join(args.matrix_dir, f"*L{layer}*.pt")
        candidates = glob.glob(pattern)

        # Simple filter: prefer files containing 'trim' or 'rema'
        files = [f for f in candidates if 'trim' in f or 'rema' in f]
        if not files and candidates: files = candidates  # Fall back to all candidates if none matched

        if not files:
            print(f"L{layer:<5} | No matrix file found.")
            continue

        target_file = files[0]  # Take the first one

        try:
            # Reconstruct matrix
            update_mat = reconstruct_matrix(target_file, device=model.device)

            # Run test
            sens_signal, sens_noise = measure_ablation_sensitivity(
                model, layer, update_mat, tokenizer, qa_pairs
            )

            # Compute signal-to-noise ratio
            ratio = sens_signal / (sens_noise + 1e-9)

            # Simple verdict
            verdict = "✅ VALID" if ratio > 1.2 else ("⚠️ WEAK" if ratio > 1.0 else "❌ NOISE")

            print(f"L{layer:<5} | {sens_signal:.4e}   | {sens_noise:.4e}   | {ratio:.2f}x            | {verdict}")

        except Exception as e:
            print(f"L{layer:<5} | Error: {e}")
            import traceback
            traceback.print_exc()


if __name__ == "__main__":
    main()