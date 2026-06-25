import sys, os
_PROJ_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _PROJ_ROOT not in sys.path:
    sys.path.insert(0, _PROJ_ROOT)

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
import numpy as np
from tqdm import tqdm
import json
import os
import argparse
from contextlib import contextmanager


# -------------------------------------------------
# 1. Utility Functions
# -------------------------------------------------
def find_subsequence(sequence, subseq):
    """Find the position of a subsequence within a long sequence."""
    L, l = len(sequence), len(subseq)
    for i in range(L - l + 1):
        if sequence[i:i + l] == subseq:
            return list(range(i, i + l))
    return None


def parse_mquake_item(item):
    """
    Parse MQUAKE data and extract both Single-hop and Multi-hop questions.
    """
    results = []

    # Basic validation
    if "questions" not in item or "requested_rewrite" not in item:
        return []
    if len(item["requested_rewrite"]) == 0:
        return []

    # Extract Subject (usually in the first entry of requested_rewrite)
    subject = item["requested_rewrite"][0].get("subject")

    # A. Extract Multi-hop reasoning
    # Target is MQUAKE's final answer (new_answer)
    multi_hop_target = item.get("new_answer")
    for q in item["questions"]:
        if q and multi_hop_target:
            results.append({
                "type": "multi_hop",
                "prompt": q,
                "subject": subject,
                "target": multi_hop_target
            })

    # B. Extract Single-hop memory
    # Target is the single-hop question answer (usually the object of the edited fact)
    if "new_single_hops" in item:
        for sh in item["new_single_hops"]:
            sh_q = sh.get("question")
            sh_a = sh.get("answer")
            if sh_q and sh_a:
                results.append({
                    "type": "single_hop",
                    "prompt": sh_q,
                    "subject": subject,
                    "target": sh_a
                })

    return results


# -------------------------------------------------
# 2. Core Hooks: Noise Injection & State Restoration
# -------------------------------------------------
@contextmanager
def add_noise_to_subject(model, token_positions, noise_std=0.1):
    """
    [Context Manager] Add Gaussian noise to specified positions at the Embedding layer.
    Used to create a "Corrupt" state without changing the Sequence Length.
    """
    hooks = []

    def noise_hook_fn(module, inp, out):
        # out is typically a (batch, seq, dim) Tensor
        # Some model outputs are tuples, take element 0
        if isinstance(out, tuple):
            tensor = out[0]
        else:
            tensor = out

        # Generate noise (keeping device and dtype consistent)
        # Note: Here we add noise to all samples in the batch, assuming batch_size=1
        noise = torch.randn_like(tensor[:, token_positions, :]) * noise_std

        # Must clone; otherwise in-place modification may error or affect gradients (even though it's no_grad)
        cloned = tensor.clone()
        cloned[:, token_positions, :] += noise

        if isinstance(out, tuple):
            return (cloned,) + out[1:]
        return cloned

    # Auto-detect the Embedding layer
    if hasattr(model, "get_input_embeddings"):
        embed_layer = model.get_input_embeddings()
    else:
        # Fallback for some models
        embed_layer = model.transformer.wte

    hooks.append(embed_layer.register_forward_hook(noise_hook_fn))
    try:
        yield
    finally:
        for h in hooks: h.remove()


@contextmanager
def patch_layer_hidden(model, layer_idx, token_positions, clean_hidden):
    """
    [Context Manager] Force-replace Hidden States back to the Clean Run state at a specified layer.
    """
    hooks = []

    def hook_fn(module, inp, out):
        if isinstance(out, tuple):
            target = out[0]
        else:
            target = out

        # Iterate over positions that need restoration (Subject positions)
        for pos in token_positions:
            if pos < target.shape[1]:
                # Force-overwrite with Clean Hidden
                target[:, pos, :] = clean_hidden[:, pos, :]
        return out

    # Adapt to different model architectures to locate the Layer
    if hasattr(model, "model") and hasattr(model.model, "layers"):  # Llama/Qwen
        block = model.model.layers[layer_idx]
    elif hasattr(model, "transformer") and hasattr(model.transformer, "h"):  # GPT-J/GPT-2
        block = model.transformer.h[layer_idx]
    else:
        raise ValueError(f"Unknown model architecture: {type(model)}")

    hooks.append(block.register_forward_hook(hook_fn))
    try:
        yield
    finally:
        for h in hooks: h.remove()


# -------------------------------------------------
# 3. Main Probe Loop (Causal Tracing)
# -------------------------------------------------
@torch.no_grad()
def run_hop_causal_probe(model, tokenizer, dataset, num_samples=50):
    device = next(model.parameters()).device
    # Get number of layers
    if hasattr(model.config, "num_hidden_layers"):
        num_layers = model.config.num_hidden_layers
    else:
        num_layers = model.config.n_layer

    results = []

    # Dynamically compute noise strength: following the ROME paper, set to 3x the Embedding standard deviation
    embeddings = model.get_input_embeddings().weight
    noise_std = (embeddings.std() * 3).item()
    print(f"[*] Calculated noise_std: {noise_std:.4f}")

    # Limit number of samples
    process_data = dataset[:num_samples] if num_samples > 0 else dataset

    for item in tqdm(process_data, desc="Processing Samples"):
        # 1. Parse sample (contains both Single and Multi hop)
        parsed_items = parse_mquake_item(item)

        for p_item in parsed_items:
            prompt = p_item["prompt"]
            subject = p_item["subject"]
            target = p_item["target"]
            hop_type = p_item["type"]  # "single_hop" or "multi_hop"

            # 2. Tokenize & locate
            # Models like Llama-3 may add special tokens; use return_offsets_mapping or manual search
            inputs = tokenizer(prompt, return_tensors="pt").to(device)
            input_ids_list = inputs.input_ids[0].tolist()

            # Find Subject Token positions
            subject_ids = tokenizer(subject, add_special_tokens=False).input_ids
            # Simple search logic
            pos = find_subsequence(input_ids_list, subject_ids)

            # If Subject not found (possibly due to tokenization leading-space issue), try with a prepended space
            if not pos:
                subject_ids = tokenizer(" " + subject, add_special_tokens=False).input_ids
                pos = find_subsequence(input_ids_list, subject_ids)

            if not pos:
                # Skip if still not found, to avoid errors
                continue

            # Get the target answer's Token ID (to observe Logit change)
            # Take the first token of the answer as the observation point
            target_ids = tokenizer(target, add_special_tokens=False).input_ids
            if len(target_ids) == 0: continue
            answer_token_id = target_ids[0]

            # -------- Step A: Clean Run (Baseline) --------
            # Run a clean pass, save Hidden States from all layers
            clean_outputs = model(**inputs, output_hidden_states=True)
            clean_hidden_cache = clean_outputs.hidden_states
            # clean_hidden_cache[i] is the output of the i-th layer (Llama typically has 33 entries: 1 Embed + 32 Layers)

            # Record baseline probability (Clean Score)
            clean_logit = clean_outputs.logits[0, -1, answer_token_id].item()

            # -------- Step B: Causal Tracing (Patching) --------
            # Iterate over every layer, attempting to "restore" the memory
            for layer_idx in range(num_layers):
                # 1. Corrupt: add noise at the Embedding layer
                with add_noise_to_subject(model, pos, noise_std=noise_std):
                    # 2. Restore: replace Subject state at layer_idx with the Clean state
                    # Note on indexing: model.layers[i] output corresponds to hidden_states[i+1]
                    with patch_layer_hidden(
                            model, layer_idx, pos, clean_hidden_cache[layer_idx + 1]
                    ):
                        # 3. Forward pass and observe result
                        patch_out = model(**inputs)
                        patch_logit = patch_out.logits[0, -1, answer_token_id].item()

                # 4. Record data
                results.append({
                    "case_id": item.get("case_id"),
                    "hop_type": hop_type,  # Key distinction: single vs multi
                    "layer": layer_idx,
                    "clean_logit": clean_logit,
                    "patched_logit": patch_logit,
                    # Restoration score could be computed as: restored = patched - corrupted (requires separate corrupt measurement)
                    # Or simply save patch_logit for later plotting/analysis
                })

            del clean_outputs, clean_hidden_cache
            torch.cuda.empty_cache()

    return results


# -------------------------------------------------
# 4. Program Entry Point
# -------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, required=True, help="HuggingFace model path")
    parser.add_argument("--dataset_path", type=str, required=True, help="Path to MQUAKE json dataset")
    parser.add_argument("--num_samples", type=int, default=50, help="Number of samples to probe")
    parser.add_argument("--save_dir", type=str, default="results/causal_probe")
    args = parser.parse_args()

    os.makedirs(args.save_dir, exist_ok=True)

    print(f"[*] Loading dataset from {args.dataset_path}...")
    with open(args.dataset_path, "r") as f:
        dataset = json.load(f)

    print(f"[*] Loading model {args.model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        torch_dtype=torch.float16,
        device_map="auto"
    )
    model.eval()

    print("[*] Starting Causal Probe...")
    results = run_hop_causal_probe(model, tokenizer, dataset, args.num_samples)


    short_name = args.model_name.split('/')[-1]
    save_path = os.path.join(args.save_dir, f"probe_results_{short_name}.pt")
    torch.save(results, save_path)

    print(f"Results saved to {save_path}")
    print(f"Total data points: {len(results)}")
