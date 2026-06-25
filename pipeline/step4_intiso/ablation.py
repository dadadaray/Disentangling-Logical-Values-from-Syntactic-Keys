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
import random
import hashlib
from pathlib import Path
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
from utils import nethook
import torch.nn.functional as F

# ==============================================================================
# Configuration & Constants
# ==============================================================================

BLIMP_PARADIGMS = [
    "principle_A_c_command", "principle_A_domain_1", "principle_A_domain_2",
    "principle_A_domain_3", "principle_A_reconstruction", "principle_A_case_1",
    "principle_A_case_2", "regular_plural_subject_verb_agreement_1",
    "regular_plural_subject_verb_agreement_2", "irregular_plural_subject_verb_agreement_1",
    "irregular_plural_subject_verb_agreement_2",
]


DEBUG_PROMPTS = {
    "GSM8K (Reasoning)": [
        "Question: Natalia sold clips to 48 of her friends in April, and then she sold half as many clips in May. How many clips did Natalia sell altogether in April and May?\nAnswer:",
        "Question: Janet has 3 times as many marbles as Arnold. If Arnold has 12 marbles, how many marbles do they have together?\nAnswer:",
        "Question: Weng earns $12 an hour for babysitting. Yesterday, she just did 50 minutes of babysitting. How much did she earn?\nAnswer:",
        "Question: James buys a jar of hot sauce for $2. He also buys 5 jars of pickles for $1.5 each. How much does he spend in total?\nAnswer:",
        "Question: A deep-sea monster rises from the bottom of the ocean. It rises 100 feet per minute. If the ocean is 2000 feet deep, how long does it take to reach the surface?\nAnswer:"
    ],
    "MATH (Symbolic)": [
        "Problem: Convert the point $(0,3)$ in rectangular coordinates to polar coordinates.  Enter your answer in the form $(r,\\theta),$ where $r > 0$ and $0 \\le \\theta < 2 \\pi.$",
        "Problem: Define\n\\[p = \\sum_{k = 1}^\\infty \\frac{1}{k^2} \\quad \\text{and} \\quad q = \\sum_{k = 1}^\\infty \\frac{1}{k^3}.\\]Find a way to write\n\\[\\sum_{j = 1}^\\infty \\sum_{k = 1}^\\infty \\frac{1}{(j + k)^3}\\]in terms of $p$ and $q.$",
        "Problem: If $f(x) = \\frac{3x-2}{x-2}$, what is the value of $f(-2) +f(-1)+f(0)$? Express your answer as a common fraction.",
        "Problem: How many positive whole-number divisors does 196 have?",
        "Problem: A regular hexagon can be divided into six equilateral triangles. If the perimeter of one of the triangles is 21 inches, what is the perimeter, in inches, of the regular hexagon?"
    ],
    #"MQuAKE (Knowledge)": [
        #"Question: Who is the head of state of the country where Ellie Kemper holds a citizenship?",
        #"Question: What is the birthplace of the person who created Tetris?",
        #"Question: What is the country of citizenship of Marc Cherry?",
        #"Question: What is the name of the current head of the Canada government?",
        #"Question: Where was LATAM Chile founded?"
    #]
}


def parse_args():
    parser = argparse.ArgumentParser(description="Final Experiment V12 (Robust)")
    parser.add_argument("--model_path", type=str, default="/data/users/yanrongen/AnyEdit/LLM-Qwen2.5-7B")
    parser.add_argument("--rema_dir", type=str, default="/data/users/yanrongen/ROME-MEMIT/rema_matrices_qwen")
    parser.add_argument("--mom2_dir", type=str, default="/data/users/yanrongen/ROME-MEMIT/mom2_eig")
    parser.add_argument("--data_root", type=str, default="/data/users/yanrongen/ROME-MEMIT/data")
    parser.add_argument("--cache_dir", type=str, default="./matrix_cache")

    parser.add_argument("--layer", type=int, default=16)
    parser.add_argument("--rema_type", type=str, default="math", choices=["math", "gsm8k"])
    parser.add_argument("--rema_k", type=int, default=64)
    parser.add_argument("--mom2_k", type=int, default=128)

    parser.add_argument("--mquake_dir", type=str, default="")
    parser.add_argument("--blimp_dir", type=str, default="")
    parser.add_argument("--tasks", nargs="+", default=["mquake", "gsm8k", "math", "blimp"])
    parser.add_argument("--probe_limit", type=int, default=100)

    parser.add_argument("--run_gsm8k_sweep", action="store_true", help="Run GSM8K Rank Sweep")
    parser.add_argument("--run_math_sweep", action="store_true", help="Run MATH Rank Sweep")
    parser.add_argument("--run_mom2_sweep", action="store_true", help="Run MOM2 Rank Sweep")
    parser.add_argument("--run_layer_sweep", action="store_true", help="Run Layer Sweep (L6, L16, L26)")
    parser.add_argument("--run_debug", action="store_true", help="Run debug")
    parser.add_argument("--run_ablation", action="store_true", help="Run Ablation Study (RandCov vs RandVec)")
    parser.add_argument("--run_capture", action="store_true", help="Run Main Figure Data Capture (Trajectory)")


    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 123, 999], help="Random seeds to run")

    return parser.parse_args()


# ==============================================================================
# Smart File Loader
# ==============================================================================
def find_file(directory, pattern_keywords, extension, strict=True):
    if not os.path.exists(directory):
        if strict:
            print(f"   [Warning] Directory not found: {directory}")
            return None  # Downgrade to Warning instead of crashing
        return None

    candidates = []
    for root, _, files in os.walk(directory):
        for file in files:
            if file.endswith(extension):
                if all(str(k).lower() in file.lower() for k in pattern_keywords):
                    candidates.append(os.path.join(root, file))

    if not candidates:
        if strict:
            # Downgrade to Warning to prevent the whole program from crashing
            print(f"   [Warning] No {extension} file found in {directory} matching {pattern_keywords}")
        return None

    candidates.sort(key=len)
    return candidates[0]


def run_debug_generation(model, tokenizer, layer, method, scale):
    """
    [Fixed] Run 15-shot sampling, auto-adapting to Llama-3 / Qwen chat templates,
    with precise extraction of the generated content.
    """
    print(f"\n      🔎 [DEBUG SAMPLE] L{layer} | {method} | Scale={scale * 100:.0f}%")
    print(f"      {'=' * 50}")

    # Get model name to select the appropriate template
    model_name = tokenizer.name_or_path.lower() if tokenizer.name_or_path else ""

    for category, prompts in DEBUG_PROMPTS.items():
        print(f"      --- {category} ---")
        for i, prompt_text in enumerate(prompts):

            # 1. Smart prompt construction (adapts to Qwen and Llama)
            if "qwen" in model_name:
                # Qwen ChatML format
                full_prompt = f"<|im_start|>user\n{prompt_text}<|im_end|>\n<|im_start|>assistant\n"
            elif "llama" in model_name or "llama3" in model_name:
                # Llama-3 format
                full_prompt = f"<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\n{prompt_text}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
            else:
                # Fallback (Base Model style)
                full_prompt = f"Question: {prompt_text}\nAnswer:"

            # 2. Encode and record the input length
            inputs = tokenizer(full_prompt, return_tensors="pt").to(model.device)
            input_len = inputs.input_ids.shape[1]  # Critical: record how long the input is

            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=512,
                    do_sample=False,  # Greedy decoding for reproducibility
                    temperature=0.0,
                    pad_token_id=tokenizer.eos_token_id
                )

            # 3. Precise truncation: only decode the newly generated tokens
            # outputs[0] contains [Input_Ids + Generated_Ids]
            generated_ids = outputs[0][input_len:]
            generated = tokenizer.decode(generated_ids, skip_special_tokens=True).strip()

            # Collapse into a single line (remove newlines to keep logs tidy)
            #display_text = generated.replace('\n', ' ').replace('\r', '')
            #if len(display_text) > 200: display_text = display_text[:200] + "..."

            print(f"\n      [Case {i + 1}]")
            print(f"      Input: {prompt_text}")
            print(f"      Gen  : {generated}")
            print(f"      {'.' * 40}")

    print(f"      {'=' * 60}\n")
# --- MQuAKE Loader ---
def load_mquake_data(data_dir, limit=500):
    print(f"Loading MQuAKE from {data_dir}...")
    path = find_file(data_dir, ["mquake"], ".json", strict=False)
    if not path: path = find_file(data_dir, ["cf"], ".json", strict=False)
    if not path: path = find_file(data_dir, ["math"], ".json", strict=False)

    if not path:
        print("   ❌ MQuAKE file not found. Skipping.")
        return []

    qa_pairs = []
    with open(path, 'r') as f:
        try:
            data = json.load(f)
        except:
            f.seek(0); data = [json.loads(line) for line in f]

    for item in data:
        if len(qa_pairs) >= limit: break
        q_list = item.get('questions', [])
        new_ans = item.get('new_answer')
        old_ans = item.get('answer')
        if q_list and new_ans and new_ans != old_ans:
            qa_pairs.append((q_list[0], new_ans))

    print(f"   Loaded {len(qa_pairs)} counterfactual reasoning pairs.")
    return qa_pairs


# --- GSM8K Loader ---
def load_gsm8k_data(data_dir, limit=500):
    print(f"Loading GSM8K from {data_dir}...")
    path = find_file(data_dir, ["gsm8k"], ".jsonl", strict=False)
    if not path: path = find_file(data_dir, ["gsm8k"], ".json", strict=False)

    if not path:
        print("   ❌ GSM8K file not found. Skipping.")
        return []

    qa_pairs = []
    with open(path, 'r') as f:
        try:  # Try JSONL first
            for line in f:
                if len(qa_pairs) >= limit: break
                item = json.loads(line)
                q, a = item.get('question'), item.get('answer')
                if a and "####" in a: a = a.split("####")[-1].strip()
                if q and a: qa_pairs.append((q, a))
        except:  # Fallback to JSON
            f.seek(0)
            data = json.load(f)
            for item in data:
                if len(qa_pairs) >= limit: break
                q, a = item.get('question'), item.get('answer')
                if q and a: qa_pairs.append((q, a))

    print(f"   Loaded {len(qa_pairs)} GSM8K pairs.")
    return qa_pairs


# --- MATH Loader ---
def load_math_data(data_dir, limit=500):
    print(f"Loading MATH from {data_dir}...")
    path = find_file(data_dir, ["math"], ".json", strict=False)
    if not path:
        print("   ❌ MATH file not found. Skipping.")
        return []

    qa_pairs = []
    with open(path, 'r') as f:
        try:
            data = json.load(f)
        except:
            f.seek(0); data = [json.loads(line) for line in f]

    for item in data:
        if len(qa_pairs) >= limit: break
        q = item.get('problem') or item.get('question')
        a = item.get('solution') or item.get('answer')
        if q and a: qa_pairs.append((q, a))

    print(f"   Loaded {len(qa_pairs)} MATH pairs.")
    return qa_pairs


# --- BLiMP Loader ---
def load_blimp_fixed(blimp_root, paradigms=BLIMP_PARADIGMS, samples_per_paradigm=1):
    print(f"Loading BLiMP from {blimp_root}...")

    if not os.path.exists(blimp_root):
        print(f"   ❌ BLiMP dir not found: {blimp_root}")
        return []

    all_pairs = []
    for paradigm in paradigms:
        try:
            pattern = os.path.join(blimp_root, f"*{paradigm}*.jsonl")
            files = glob.glob(pattern)
            if not files:
                pattern = os.path.join(blimp_root, f"*{paradigm}*.json")
                files = glob.glob(pattern)
            if not files: continue

            target_file = files[0]
            pairs_loaded = 0
            with open(target_file, "r", encoding="utf-8") as f:
                for line in f:
                    if not line.strip(): continue
                    try:
                        data = json.loads(line)
                        if 'sentence_good' in data and 'sentence_bad' in data:
                            all_pairs.append((data['sentence_good'], data['sentence_bad']))
                            pairs_loaded += 1
                        if pairs_loaded >= samples_per_paradigm: break
                    except:
                        continue
        except Exception as e:
            print(f"   Error loading {paradigm}: {e}")

    print(f"   Loaded Total: {len(all_pairs)} BLiMP pairs.")
    return all_pairs


# ==============================================================================
# Core Matrix Reconstruction (V9: Robust NPZ Support)
# ==============================================================================

import gc  # Ensure gc is imported
from contextlib import contextmanager


@contextmanager
def temporary_parameter(model, param_name, new_value):
    """
    A context manager that temporarily modifies a model parameter,
    automatically restoring the original value upon exit.
    Serves as a replacement for the non-existent nethook.set_parameter.
    """
    # 1. Get the parameter object
    # nethook.get_parameter returns an nn.Parameter object
    param = nethook.get_parameter(model, param_name)

    # 2. Back up the original data (clone to avoid reference mutation)
    old_data = param.data.clone()

    # 3. Apply the new weight
    # Note: must modify the .data attribute and ensure matching dtype/device
    param.data = new_value.to(param.device, dtype=param.dtype)

    try:
        yield
    finally:
        # 4. Restore the original data (whether or not an error occurred)
        param.data = old_data
        # print(f"   🔄 Restored parameter: {param_name}")


def extract_top_k_components(data, k, device):

    tensor = None

    # 1. Try to extract from standard keys first
    if isinstance(data, dict):
        priority_keys = ['projection_matrix', 'U', 'u', 'eigenvectors', 'eigvecs', 'mom2']
        for key in priority_keys:
            if key in data:
                tensor = data[key]
                break

        # 2. Fallback search: look for a large matrix matching model dimensions
        if tensor is None:
            # Qwen=3584/18944, Llama=4096/14336
            valid_dims = {3584, 18944, 4096, 14336}
            for key, v in data.items():
                if isinstance(v, torch.Tensor):
                    if any(d in v.shape for d in valid_dims):
                        tensor = v
                        break

    elif isinstance(data, torch.Tensor):
        tensor = data

    # 3. Error interception
    if tensor is None:
        found_keys = list(data.keys()) if isinstance(data, dict) else "N/A"
        raise ValueError(
            f"\n❌ CRITICAL ERROR: Could not find a valid matrix in the file.\n"
            f"   File contains Keys: {found_keys}\n"
            f"   Reason: No Tensor matches Qwen (3584) or Llama (4096) dimensions.\n"
            f"   👉 Check: Did you read a 'spectral_data' (spectral analysis result) instead of a matrix file?"
        )

    # 4. Prepare
    del data
    gc.collect()
    torch.cuda.empty_cache()

    shape = tensor.shape
    dim_idx = -1

    # --- Dimension heuristics (Qwen-compatible) ---
    # Qwen Hidden (3584) or Llama Hidden (4096)
    if (3584 in shape or 4096 in shape) and (18944 not in shape and 14336 not in shape):
        target = 3584 if 3584 in shape else 4096
        dim_idx = 0 if shape[0] == target else 1

    # Qwen Inter (18944) or Llama Inter (14336)
    elif (18944 in shape or 14336 in shape) and (3584 not in shape and 4096 not in shape):
        target = 18944 if 18944 in shape else 14336
        dim_idx = 0 if shape[0] == target else 1

    elif shape[0] == shape[1]:  # Square matrix
        dim_idx = -1

    # --- Extract ---
    if dim_idx != -1:
        if dim_idx == 0:
            current_k = min(tensor.shape[1], k)
            result = tensor[:, :current_k]
        else:
            current_k = min(tensor.shape[0], k)
            result = tensor[:current_k, :].T
        return result.to(device, dtype=torch.float32)

    # --- Square matrix decomposition (Fallback) ---
    try:
        try:
            target_dtype = torch.bfloat16
            tensor_gpu = tensor.to(device, dtype=target_dtype)
        except:
            target_dtype = torch.float16
            tensor_gpu = tensor.to(device, dtype=target_dtype)

        tensor_gpu += torch.eye(shape[0], device=device, dtype=target_dtype) * 1e-4
        vals, vecs = torch.linalg.eigh(tensor_gpu)
        result = vecs[:, -k:]  # Take the top-k eigenvectors

        del tensor_gpu, vals, vecs
        torch.cuda.empty_cache()
        return result.float()

    except Exception as e:
        print(f"   ⚠️ GPU Decomposition failed ({e}), falling back to CPU...")
        torch.cuda.empty_cache()
        tensor_cpu = tensor.float().cpu()
        tensor_cpu += torch.eye(shape[0]) * 1e-4
        try:
            vals, vecs = torch.linalg.eigh(tensor_cpu)
            result = vecs[:, -k:]
        except:
            U, S, V = torch.linalg.svd(tensor_cpu)
            result = U[:, :k]
        return result.to(device)


def reconstruct_matrix_from_source(path, k, device, target_shape):
    if path.endswith(".npz"):
        import numpy as np
        data_np = np.load(path)
        data = {k: torch.from_numpy(v) for k, v in data_np.items() if v.dtype.kind in {'f', 'i'}}
    else:
        data = torch.load(path, map_location=device)

    V_comp = extract_top_k_components(data, k, device)
    V_comp = F.normalize(V_comp, dim=0)

    torch.manual_seed(42)
    dim, k_actual = V_comp.shape
    if dim == target_shape[0]:
        U = V_comp
        V_rand = torch.randn(target_shape[1], k_actual, device=device)
        return U @ F.normalize(V_rand, dim=0).T
    elif dim == target_shape[1]:
        V_in = V_comp
        U_rand = torch.randn(target_shape[0], k_actual, device=device)
        return F.normalize(U_rand, dim=0) @ V_in.T
    else:
        raise ValueError(f"Dimension mismatch {dim}")


def get_hybrid_matrix_with_cache(cache_dir, args, device, target_shape):
    os.makedirs(cache_dir, exist_ok=True)
    cache_filename = f"hybrid_{args.rema_type}_wiki_L{args.layer}_kR{args.rema_k}_kM{args.mom2_k}.pt"
    cache_path = os.path.join(cache_dir, cache_filename)

    if os.path.exists(cache_path):
        try:
            mat = torch.load(cache_path, map_location=device)
            if mat.shape == target_shape: return mat
        except:
            pass

    # DEBUG print: check what raw ingredients the Hybrid matrix is actually using
    print(f"\n   🔎 [DEBUG] Constructing Hybrid Matrix:")
    print(f"      Target Shape: {target_shape}")

    # 1. Load REMA (U)
    rema_path = find_file(args.rema_dir, ["rema", args.rema_type, f"L{args.layer}"], ".pt")
    print(f"      REMA Path (U): {rema_path}")

    if not rema_path:
        raise FileNotFoundError(f"REMA file not found in {args.rema_dir}")

    # Load and check U dimensions
    raw_rema = torch.load(rema_path, map_location=device)
    U = extract_top_k_components(raw_rema, args.rema_k, device)
    print(f"      Raw REMA Dim : {U.shape} (Should contain 3584 or 4096)")

    expected_out_dim = target_shape[0]  # Qwen=3584
    if U.shape[0] != expected_out_dim: U = U.T

    if U.shape[0] != expected_out_dim:
        print(f"      ❌ REMA Dim Error: Got {U.shape[0]}, Expected {expected_out_dim}")

    # 2. Load MOM2 (V)
    mom2_path = find_file(args.mom2_dir, [f"layers.{args.layer}", "mom2"], ".npz", strict=False) or \
                find_file(args.mom2_dir, [str(args.layer)], ".pt", strict=True)
    print(f"      MOM2 Path (V): {mom2_path}")

    if mom2_path.endswith(".npz"):
        import numpy as np
        d = np.load(mom2_path)
        data_mom2 = {k: torch.from_numpy(v) for k, v in d.items() if v.dtype.kind in {'f', 'i'}}
    else:
        data_mom2 = torch.load(mom2_path, map_location=device)

    V = extract_top_k_components(data_mom2, args.mom2_k, device)
    print(f"      Raw MOM2 Dim : {V.shape} (Should contain 18944 or 14336)")

    expected_in_dim = target_shape[1]  # Qwen=18944
    if V.shape[0] != expected_in_dim and V.shape[1] == expected_in_dim:
        V = V.T

    if V.shape[0] != expected_in_dim:
        raise ValueError(f"Hybrid Matrix Error: MOM2 V shape {V.shape} invalid. Expected rows={expected_in_dim}.")

    # Rank Truncation
    rank = min(U.shape[1], V.shape[1])
    print(f"      -> Hybrid Rank Truncation: Using top-{rank} components")

    U_top = U[:, :rank]
    V_top = V[:, :rank]

    mat = U_top @ V_top.T
    torch.save(mat, cache_path)
    return mat


def get_matrix_with_cache(cache_dir, method, identifier, layer, k, device, target_shape, source_path_finder):
    os.makedirs(cache_dir, exist_ok=True)
    cache_filename = f"{method.lower()}_{identifier}_L{layer}_k{k}.pt"
    cache_path = os.path.join(cache_dir, cache_filename)

    if os.path.exists(cache_path):
        try:
            mat = torch.load(cache_path, map_location=device)
            if mat.shape == target_shape:
                return mat
            else:
                print(f"   ⚠️ Cache shape mismatch: {mat.shape} vs target {target_shape}. Recomputing...")
        except:
            pass

    try:
        source_path = source_path_finder()

        # DEBUG print: check which file REMA is actually reading from
        print(f"\n   🔎 [DEBUG] Loading {method} Matrix:")
        print(f"      Target Shape : {target_shape} (Model Layer)")
        print(f"      Source Path  : {source_path}")

        if not source_path:
            raise FileNotFoundError(f"Source file for {method} not found.")

        mat = reconstruct_matrix_from_source(source_path, k, device, target_shape)
        torch.save(mat, cache_path)
        return mat
    except Exception as e:
        # Print a more detailed error stack trace
        import traceback
        print(f"   ❌ Error {method}: {e}")
        # traceback.print_exc()  # Uncomment this line if you need the code line numbers
        return torch.randn(target_shape, device=device)


import re
from collections import Counter
import numpy as np


# [Keep your extract_answer unchanged] ...

def calculate_fluency(text):
    """
    Compute text fluency (based on weighted N-gram entropy).
    High Score = Rich/Diverse text
    Low Score (near 0) = Repetitive/Collapsed text
    """
    text = text.strip()
    if not text:
        return 0.0

    # Simple tokenization logic
    tokens = re.findall(r"\w+|[^\w\s]", text, re.UNICODE)

    if len(tokens) < 2:
        return 0.0

    def compute_entropy(n):
        if len(tokens) < n:
            return 0.0

        ngrams = [tuple(tokens[i:i + n]) for i in range(len(tokens) - n + 1)]
        if not ngrams:
            return 0.0

        counts = Counter(ngrams)
        total = sum(counts.values())

        # sum(p * log2(p))
        probs = np.array(list(counts.values())) / total
        entropy = -np.sum(probs * np.log2(probs))
        return entropy

    e2 = compute_entropy(2)
    e3 = compute_entropy(3)

    # Weighted scoring logic
    weighted_e2 = e2 * (2 / 3)
    weighted_e3 = e3 * (4 / 3)

    score = np.mean([weighted_e2, weighted_e3])

    return score


def eval_generation_metrics(model, tokenizer, data, task_name, limit=500, batch_size=32):
    """
    [Fixed] Eliminate args-related errors, add debug printing, smart template matching.
    """
    correct = 0
    total_fluency = 0.0
    count = 0

    # 1. Truncate data
    test_data = data[:limit]

    # 2. Tokenizer setup
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = 'left'  # Decoder must use left-padding

    # 3. Iterate by batch
    for i in tqdm(range(0, len(test_data), batch_size), desc=f"   [Gen:{task_name}]", leave=False):
        batch = test_data[i: i + batch_size]

        # 3.1 Build prompts in batch
        prompts = []
        golds = []

        # Fix 1: Get model name from tokenizer, no need for args
        model_id = tokenizer.name_or_path.lower() if tokenizer.name_or_path else ""

        for q, a in batch:
            # Try the standard chat template
            try:
                messages = [{"role": "user", "content": q}]
                full_prompt = tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True
                )
            except:
                # Fix 2: Manual fallback (Qwen uses ChatML)
                if "qwen" in model_id:
                    full_prompt = f"<|im_start|>user\n{q}<|im_end|>\n<|im_start|>assistant\n"
                elif "llama" in model_id:
                    full_prompt = f"<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\n{q}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
                else:
                    full_prompt = f"Question: {q}\nAnswer:"

            prompts.append(full_prompt)
            golds.append(str(a).lower().strip().replace(",", ""))

        # 3.2 Batch encode
        inputs = tokenizer(prompts, return_tensors="pt", padding=True, truncation=True).to(model.device)
        input_len = inputs.input_ids.shape[1]

        eos_ids = [tokenizer.eos_token_id]  # Base EOS

        # Add extra stop tokens depending on model type
        if "qwen" in model_id:
            eos_ids.append(tokenizer.convert_tokens_to_ids("<|im_end|>"))
        elif "llama-3" in model_id or "llama3" in model_id:
            eos_ids.append(tokenizer.convert_tokens_to_ids("<|eot_id|>"))

        # Clean the list: filter out None and non-integer entries
        valid_eos_ids = [tid for tid in eos_ids if tid is not None and isinstance(tid, int)]

        generate_kwargs = {
            "max_new_tokens": 512,
            "do_sample": False,
            "temperature": 0.0,
            "pad_token_id": tokenizer.eos_token_id
        }
        if valid_eos_ids:
            generate_kwargs["eos_token_id"] = valid_eos_ids

        # 3.3 Batch generation
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=512,
                do_sample=False,
                temperature=0.0,
                pad_token_id=tokenizer.eos_token_id,
                # Fix 3: Explicitly specify EOS tokens to prevent Qwen from failing to stop
                eos_token_id=[tokenizer.eos_token_id, tokenizer.convert_tokens_to_ids("<|im_end|>")]
            )

        # 3.4 Batch decode and evaluate
        for j, output in enumerate(outputs):
            gen_tokens = output[input_len:]
            gen_text = tokenizer.decode(gen_tokens, skip_special_tokens=True).strip()

            # DEBUG core: Print the first few generated texts to see what they look like
            #if count < 3:
                #print(f"\n   🔎 [DEBUG Gen] Prompt Tail: ...{prompts[j][-50:].replace(chr(10), ' ')}")
                #print(f"   🔎 [DEBUG Gen] Output: {gen_text}")
                #print(f"   🔎 [DEBUG Gen] Gold:   {golds[j]}")

            # 1. Acc
            pred = extract_answer(gen_text, task_name)

            # MQuAKE special handling
            if task_name.lower() == 'mquake':
                if golds[j] in gen_text.lower(): correct += 1
            else:
                if pred == golds[j]: correct += 1

            # 2. Fluency
            total_fluency += calculate_fluency(gen_text)
            count += 1

    return {
        "acc": (correct / count) * 100 if count > 0 else 0,
        "flu": (total_fluency / count) if count > 0 else 0
    }

def extract_answer(text, task_type):
    """Extract the core answer from generated text."""
    text = text.split("assistant")[-1].strip().lower()  # Only process the assistant portion
    if task_type.lower() == 'gsm8k':
        # Match the last number, supporting comma thousands separators
        match = re.findall(r"(\d+(?:,\d+)?(?:\.\d+)?)", text)
        if match:
            return match[-1].replace(",", "")
    elif task_type.lower() == 'math':
        # Prefer \boxed{...}
        match = re.search(r"\\boxed\{(.*?)\}", text)
        if match: return match.group(1).strip()
        # Fallback: take the last number
        nums = re.findall(r"(\d+(?:\.\d+)?)", text)
        return nums[-1] if nums else None
    elif task_type.lower() == 'mquake':
        return text  # MQuAKE typically uses a direct string containment check
    return None


# ==============================================================================
# Additional Utility Functions
# ==============================================================================

def run_baseline_robustness(model, tokenizer, datasets, seeds=[42, 123, 999], limit=200):
    """
    Measure the original model's accuracy on 200 questions across three seeds.
    """
    print(f"\n{'#' * 80}")
    print(f"📊 Baseline Robustness Check (Original Model)")
    print(f"{'#' * 80}")

    results = {}

    for seed in seeds:
        print(f"\n>>> Running Baseline with Seed {seed}...")
        torch.manual_seed(seed)
        random.seed(seed)
        if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)

        for name, data in datasets.items():
            # Use eval_generation_metrics for accuracy and fluency evaluation
            metrics = eval_generation_metrics(model, tokenizer, data, name, limit=limit, batch_size=32)
            print(f"   {name:<8}: Acc={metrics['acc']:>5.2f}% | Flu={metrics['flu']:>4.2f}")

            if name not in results: results[name] = []
            results[name].append(metrics['acc'])

    print(f"\n{'-' * 40}")
    print(f"✅ Baseline Summary (Mean ± Std over {len(seeds)} seeds):")
    for name, accs in results.items():
        mean_acc = np.mean(accs)
        std_acc = np.std(accs)
        print(f"   {name:<8}: {mean_acc:.2f}% ± {std_acc:.2f}")
    print(f"{'-' * 40}\n")


def collect_tsne_data(model, tokenizer, datasets, layer, limit=200, output_dir="tsne_data"):
    """
    Utility function for collecting data to produce t-SNE plots.
    """
    os.makedirs(output_dir, exist_ok=True)
    print(f"\n🎨 Collecting t-SNE Data (L{layer})...")

    all_records = []

    for label, data in datasets.items():
        print(f"   -> Processing {label} ({len(data[:limit])} samples)...")

        # Clean data and extract text
        clean_texts = []
        for item in data[:limit]:
            if isinstance(item, tuple):
                clean_texts.append(item[0])  # Take the Question or Sentence
            else:
                clean_texts.append(item)

        # Reuse the existing get_hidden_states function
        # Note: get_hidden_states returns a numpy array [N, Hidden_Dim]
        states = get_hidden_states(model, tokenizer, clean_texts, layer, limit=limit)

        # Package the data
        for i, vec in enumerate(states):
            all_records.append({
                "label": label,
                "text_snippet": clean_texts[i][:50],
                "vector": vec  # numpy array
            })

    save_path = os.path.join(output_dir, f"tsne_L{layer}_data.pt")
    torch.save(all_records, save_path)
    print(f"✅ Saved {len(all_records)} records to {save_path}")


def capture_trajectory_states(model, tokenizer, prompts, layer, label):

    states = get_hidden_states(model, tokenizer, prompts, layer)
    return {
        "label": label,
        "vectors": states  # numpy array [N, d]
    }


def run_experiment_step(model, tokenizer, datasets, base_scores, args, layer, rema_type, rema_k, mom2_k, seed,
                        target_methods=None, target_scales=None, verbose=False):
    """
    Encapsulate a single experiment run.
    New parameter target_methods: list, specify which methods to run (for deduplication).
    """
    torch.manual_seed(seed)
    random.seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)

    print(f"\n{'=' * 60}")
    print(f"🚀 Exp: L{layer} | REMA={rema_type}(k={rema_k}) | MOM2(k={mom2_k}) | Seed={seed}")
    print(f"{'=' * 60}")

    layer_name = f"model.layers.{layer}.mlp.down_proj.weight"
    param = nethook.get_parameter(model, layer_name)
    target_shape = param.shape
    w_down = param.data.to(model.device).float()

    matrices = {}

    def find_rema():
        # Enhanced search that explicitly excludes spectral analysis files
        if not os.path.exists(args.rema_dir): return None

        candidates = []
        for root, _, files in os.walk(args.rema_dir):
            for file in files:
                # Must contain rema, math/gsm8k, L16 keywords
                if file.endswith(".pt") and all(str(k) in file for k in ["rema", rema_type, f"L{layer}"]):

                    # Core fix: If you've generated spectral analysis plots before,
                    # the folder will contain spectral files.
                    # They must be skipped, otherwise we'd read a small file with
                    # only eigenvalues and trigger an error.
                    if "spectral" in file:
                        continue

                    candidates.append(os.path.join(root, file))

        if not candidates: return None
        # Sort by file name length; usually the shortest one is the source file
        candidates.sort(key=len)
        return candidates[0]

    def find_mom2_wrapper():
        path = find_file(args.mom2_dir, [f"layers.{layer}", "mom2"], ".npz", strict=False)
        if path: return path
        return find_file(args.mom2_dir, [f"L{layer}"], ".pt", strict=True)

    # --- 1. Load matrices on demand (save I/O) ---
    # If target_methods is specified, only load the required matrices.
    # Otherwise, default to running all methods.
    methods_to_run = target_methods if target_methods else ['Random', 'MOM2_Rand', 'MOM2_Self', 'REMA', 'Hybrid']

    # REMA / Hybrid need the REMA matrix
    if 'REMA' in methods_to_run or 'Hybrid' in methods_to_run:
        matrices['REMA'] = get_matrix_with_cache(args.cache_dir, "rema", rema_type, layer, rema_k,
                                                 model.device, target_shape, find_rema)

    # MOM2 / Hybrid need the MOM2 matrix
    if any(m in methods_to_run for m in ['MOM2_Rand', 'MOM2_Self', 'Hybrid']):
        mom2_path = find_mom2_wrapper()
        if mom2_path and mom2_path.endswith(".npz"):
            d = np.load(mom2_path)
            data_mom2 = {k: torch.from_numpy(v) for k, v in d.items() if v.dtype.kind in {'f', 'i'}}
        else:
            data_mom2 = torch.load(mom2_path, map_location='cpu')  # Load to CPU first

        V_mom2_raw = extract_top_k_components(data_mom2, mom2_k, model.device)
        del data_mom2;
        gc.collect();
        torch.cuda.empty_cache()

        if V_mom2_raw.shape[0] != target_shape[1] and V_mom2_raw.shape[1] == target_shape[1]:
            V_mom2_raw = V_mom2_raw.T

        if 'MOM2_Rand' in methods_to_run:
            matrices['MOM2_Rand'] = get_matrix_with_cache(args.cache_dir, "mom2", "wiki", layer, mom2_k,
                                                          model.device, target_shape, find_mom2_wrapper)

        if 'MOM2_Self' in methods_to_run:
            target_in_dim = w_down.shape[1]
            if V_mom2_raw.shape[0] == target_in_dim:
                v_wiki = V_mom2_raw[:, 0]
            elif V_mom2_raw.shape[1] == target_in_dim:
                v_wiki = V_mom2_raw[0, :]
            else:
                v_wiki = torch.randn(target_in_dim, device=model.device)
            v_wiki_dev = v_wiki.to(model.device).float()
            u_wiki = w_down @ v_wiki_dev
            matrices['MOM2_Self'] = F.normalize(u_wiki, dim=0).unsqueeze(1) @ F.normalize(v_wiki_dev, dim=0).unsqueeze(
                0)

    # Hybrid needs both
    if 'Hybrid' in methods_to_run:
        _orig_args = (args.rema_type, args.rema_k, args.mom2_k)
        args.rema_type, args.rema_k, args.mom2_k = rema_type, rema_k, mom2_k
        try:
            matrices['Hybrid'] = get_hybrid_matrix_with_cache(args.cache_dir, args, model.device, target_shape)
        except Exception as e:
            print(f"   ❌ Hybrid failed: {e}")
        args.rema_type, args.rema_k, args.mom2_k = _orig_args

    if 'Random' in methods_to_run:
        matrices['Random'] = torch.randn(target_shape, device=model.device)

    if 'Ablation_RandCov' in methods_to_run or 'Ablation_RandVec' in methods_to_run:
        # 1. Load the real REMA U (4096, k)
        rema_path = find_file(args.rema_dir, ["rema", rema_type, f"L{layer}"], ".pt")
        if rema_path:
            d_r = torch.load(rema_path, map_location=model.device)
            U_real = extract_top_k_components(d_r, rema_k, model.device)
            if U_real.shape[0] != target_shape[0]: U_real = U_real.T
            U_real = U_real[:, :rema_k]
        else:
            U_real = torch.randn(target_shape[0], rema_k, device=model.device)  # Fallback

        # 2. Load the real MOM2 V (14336, k)
        mom2_path = find_mom2_wrapper()
        if mom2_path:
            if mom2_path.endswith(".npz"):
                d = np.load(mom2_path)
                d_m = {k: torch.from_numpy(v) for k, v in d.items() if v.dtype.kind in {'f', 'i'}}
            else:
                d_m = torch.load(mom2_path, map_location=model.device)
            V_real = extract_top_k_components(d_m, mom2_k, model.device)

            # 1. Strict transpose logic: only transpose when dim 1 is the target dimension
            if V_real.shape[0] != target_shape[1] and V_real.shape[1] == target_shape[1]:
                V_real = V_real.T

            # 2. Enforce dimension check: prevent shapes like [128, 128] from slipping through
            if V_real.shape[0] != target_shape[1]:
                print(
                    f"   ⚠️ Warning: V_real shape mismatch {V_real.shape}, expected dim0={target_shape[1]}. Replacing with Randn.")
                V_real = torch.randn(target_shape[1], mom2_k, device=model.device)

            V_real = V_real[:, :mom2_k]
        else:
            V_real = torch.randn(target_shape[1], mom2_k, device=model.device)  # Fallback

        # Ensure consistent rank
        rank = min(U_real.shape[1], V_real.shape[1])
        U_real = U_real[:, :rank]
        V_real = V_real[:, :rank]

        # --- Construct Ablation Matrices ---

        # Case A: MOM2_Rand (Random Covariance) -> Real REMA @ Random V
        # Hypothesis: Non-random syntactic noise must be removed (Random V represents random noise directions)
        if 'Ablation_RandCov' in methods_to_run:
            V_rand = torch.randn_like(V_real)
            # Orthogonalize to mimic a real basis
            V_rand, _ = torch.linalg.qr(V_rand)
            matrices['Ablation_RandCov'] = U_real @ V_rand.T

        # Case B: Directionless REMA (Random Vector) -> Random U @ Real MOM2
        # Hypothesis: There must be a clear reasoning direction (Random U represents no direction)
        if 'Ablation_RandVec' in methods_to_run:
            U_rand = torch.randn_like(U_real)
            U_rand, _ = torch.linalg.qr(U_rand)
            matrices['Ablation_RandVec'] = U_rand @ V_real.T

    # Normalize
    for k, v in matrices.items(): matrices[k] = v / (v.norm() + 1e-8)

    # --- Eval Loop ---
    # Debug mode runs only a few scales; full experiments run all scales
    #scales = [0.05]
    if target_scales:
        scales = target_scales
    else:
        scales = [0.05, 0.10, 0.15]
    weight_norm = param.norm().item()

    # Print table header
    print(f"{'-' * 110}")
    headers = [f"{k[:6]}" for k in datasets.keys()]
    print(f"Scale  | Method   | " + " | ".join([f"{h:<20}" for h in headers]))
    print(f"{'-' * 110}")

    for scale in scales:
        epsilon = weight_norm * scale
        for method in methods_to_run:
            if method not in matrices: continue
            mat = matrices[method]
            current_epsilon = -epsilon if (method in ['Hybrid', 'REMA'] and rema_type == 'gsm8k') else epsilon
            delta = (mat * current_epsilon).to(param.dtype)

            param.data += delta  # Apply

            if verbose:
                print(f"\n   👀 [VERBOSE] Generating examples for {method} @ Scale {scale}...")
                # Call the run_debug_generation defined at the top of this file.
                # It prints the 5 GSM8K and 5 MATH examples defined in DEBUG_PROMPTS.
                run_debug_generation(model, tokenizer, layer, method, scale)

            results_str = []
            for name, data in datasets.items():
                if name == 'BLiMP':
                    res = f"{eval_accuracy(model, tokenizer, data):.2f}%"
                else:
                    # Change: use args.probe_limit and batch_size=16 (for speed)
                    # If you have large GPU memory, consider changing batch_size to 32
                    gen_metrics = eval_generation_metrics(model, tokenizer, data, name, limit=args.probe_limit,
                                                          batch_size=32)
                    curr_lp = eval_logprob(model, tokenizer, data)
                    lp_delta = curr_lp - base_scores[name]
                    # Acc / Fluency / LogProb
                    res = f"A:{gen_metrics['acc']:>3.2f}%|F:{gen_metrics['flu']:>4.2f}|L:{lp_delta:+.2f}"
                results_str.append(f"{res:<20}")

            param.data -= delta  # Restore
            print(f"{scale * 100:>4.1f}% | {method:<8} | " + " | ".join(results_str))

    print(f"{'-' * 110}\n")
    del matrices;
    gc.collect();
    torch.cuda.empty_cache()


def run_main_figure_capture(model, tokenizer, datasets, args, layer=16, seed=42):
    """
    Main Figure Data Collector (Experimental Replica Version).
    Strictly replicates the injection logic of run_experiment_step:
    1. Target: mlp.down_proj (4096 -> 14336)
    2. MOM2 Matrix: forced identification as 14336-dimensional (V)
    3. REMA Matrix: identified as 4096-dimensional (U)
    """
    print(f"\n{'=' * 80}")
    print(f"🎨 RUNNING MAIN FIGURE CAPTURE (Layer {layer}, Seed {seed})")
    print(f"{'=' * 80}")

    # ==========================================================================
    # 1. Prepare Prompts
    # ==========================================================================
    capture_limit = 200
    target_tasks_lower = ['gsm8k', 'math']
    dataset_lookup = {k.lower(): v for k, v in datasets.items()}
    prompts_dict = {}

    print("   -> Preparing prompts...")
    for task in target_tasks_lower:
        if task in dataset_lookup:
            raw_data = dataset_lookup[task][:capture_limit]
            prompts_dict[task] = [item[0] if isinstance(item, tuple) else item for item in raw_data]
            print(f"      - {task}: {len(prompts_dict[task])} samples")
        else:
            print(f"   ⚠️ Warning: Dataset '{task}' not found, skipping.")

    if not prompts_dict:
        print("❌ No datasets found! Aborting.")
        return

    capture_results = {task: {} for task in prompts_dict.keys()}

    # ==========================================================================
    # 2. State A: Original Model (Base)
    # ==========================================================================
    print("\n📸 Capturing State: [Original / Base]...")
    for task, prompts in prompts_dict.items():
        vecs = get_hidden_states(model, tokenizer, prompts, layer)
        capture_results[task]['Base'] = vecs

    # ==========================================================================
    # Matrix Setup
    # ==========================================================================
    mom2_k = 128
    rema_k = 16
    rema_type = 'math'
    scale = 0.1

    # Return to the original experiment target: down_proj
    layer_name = f"model.layers.{layer}.mlp.down_proj"

    try:
        W_old = nethook.get_parameter(model, f"{layer_name}.weight")
        print(f"   -> Target Weight ({layer_name}): {W_old.shape}")  # Should be [4096, 14336]
        target_in_dim = W_old.shape[1]  # 14336
    except LookupError:
        print(f"❌ Layer {layer_name} not found.")
        return

    # ==========================================================================
    # (A) Load MOM2 (V) - Must be 14336-dimensional
    # ==========================================================================
    print(f"\n   -> 🔍 Locating MOM2 Matrix (k={mom2_k})...")
    search_strategies = [["mom2", str(layer)], ["wikipedia", f"L{layer}"], ["wiki", f"L{layer}"], [f"L{layer}"]]
    mom2_path = None
    for kws in search_strategies:
        mom2_path = find_file(args.mom2_dir, kws, ".pt", strict=False) or find_file(args.mom2_dir, kws, ".npz",
                                                                                    strict=False)
        if mom2_path: break

    if not mom2_path: return

    try:
        if mom2_path.endswith('.npz'):
            d_m = np.load(mom2_path)
            raw_mom2 = list(d_m.values())[0]
            raw_mom2 = torch.from_numpy(raw_mom2)
        else:
            raw_mom2 = torch.load(mom2_path, map_location=model.device)

        # Extract k components
        V_mom2 = extract_top_k_components(raw_mom2, mom2_k, model.device)

        # Key: Ensure V is [14336, k]
        # If it's [k, 14336], transpose.
        # If it's [4096, k], we loaded the wrong matrix (Hidden) and it cannot be applied to down_proj.
        if V_mom2.shape[0] != target_in_dim and V_mom2.shape[1] == target_in_dim:
            V_mom2 = V_mom2.T

        if V_mom2.shape[0] != target_in_dim:
            print(
                f"   ❌ MOM2 Dimension Mismatch! Expected {target_in_dim}, got {V_mom2.shape[0]}. Cannot apply to down_proj.")
            return

        V_mom2 = V_mom2[:, :mom2_k].float()
        print(f"      V_mom2 Shape: {V_mom2.shape} (Correct for down_proj)")

    except Exception as e:
        print(f"   ❌ Error loading MOM2: {e}")
        return

    # ==========================================================================
    # (B) Load REMA (U) - Must be 4096-dimensional
    # ==========================================================================
    print(f"   -> 🔍 Locating REMA Matrix (Type={rema_type}, k={rema_k})...")
    rema_path = find_file(args.rema_dir, ["rema", rema_type, f"L{layer}"], ".pt", strict=False)
    if not rema_path: return

    try:
        raw_rema = torch.load(rema_path, map_location=model.device)
        U_rema = extract_top_k_components(raw_rema, rema_k, model.device)
        # Ensure U is [4096, k]
        if U_rema.shape[0] != 4096 and U_rema.shape[1] == 4096: U_rema = U_rema.T
        U_rema = U_rema[:, :rema_k].float()
        print(f"      U_rema Shape: {U_rema.shape} (Correct for output dim)")
    except Exception as e:
        print(f"   ❌ Error loading REMA: {e}")
        return

    # ==========================================================================
    # 3. State B: MOM2_Self (Injection)
    # ==========================================================================
    print(f"\n📸 Capturing State: [MOM2_Self] (scale={scale})...")

    # Logic: Delta = scale * (W @ V) @ V.T
    # [4096, 14336] @ [14336, k] -> [4096, k]
    # [4096, k] @ [k, 14336] -> [4096, 14336]
    # This produces a delta that can be added to the weight matrix

    W_V = W_old.to(torch.float32) @ V_mom2
    delta_mom2 = scale * (W_V @ V_mom2.T)
    delta_mom2 = delta_mom2.to(dtype=W_old.dtype, device=W_old.device)

    with temporary_parameter(model, f"{layer_name}.weight", W_old + delta_mom2):
        for task, prompts in prompts_dict.items():
            capture_results[task]['MOM2_Self'] = get_hidden_states(model, tokenizer, prompts, layer)

    # ==========================================================================
    # 4. State C: Hybrid (Ours)
    # ==========================================================================
    print(f"\n📸 Capturing State: [Hybrid] (scale={scale})...")

    min_rank = min(V_mom2.shape[1], U_rema.shape[1])
    U_h = U_rema[:, :min_rank]
    V_h = V_mom2[:, :min_rank]

    eff_scale = -scale if rema_type == 'gsm8k' else scale

    # Logic: Delta = scale * (U @ V.T)
    # [4096, k] @ [k, 14336] -> [4096, 14336]
    delta_hybrid = eff_scale * (U_h @ V_h.T)
    delta_hybrid = delta_hybrid.to(dtype=W_old.dtype, device=W_old.device)

    with temporary_parameter(model, f"{layer_name}.weight", W_old + delta_hybrid):
        for task, prompts in prompts_dict.items():
            capture_results[task]['Hybrid'] = get_hidden_states(model, tokenizer, prompts, layer)

    # ==========================================================================
    # 5. Saving
    # ==========================================================================
    save_dir = getattr(args, 'output_dir', 'results')
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, f"main_figure_vectors_L{layer}.pt")

    print(f"   💾 Saving to: {os.path.abspath(save_path)}")
    torch.save(capture_results, save_path)
    print(f"\n✅ Main Figure Data Saved Successfully!")
# ==============================================================================
# Eval Wrappers
# ==============================================================================
def eval_logprob(model, tokenizer, data):
    total = 0;
    count = 0
    for q, a in data:
        prompt = f"<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\n{q}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
        full = prompt + a

        inputs = tokenizer(full, return_tensors="pt").to(model.device)
        labels = inputs.input_ids.clone()

        prompt_len = tokenizer(prompt, return_tensors="pt").input_ids.shape[1]

        if prompt_len > labels.shape[1]: prompt_len = labels.shape[1]

        labels[:, :prompt_len] = -100

        with torch.no_grad():
            try:
                out = model(**inputs, labels=labels)
                total += -out.loss.item()
                count += 1
            except Exception as e:
                print(f"[Eval Error] {e}")
                continue

    return total / (count + 1e-9)


def eval_accuracy(model, tokenizer, data):
    if not data: return 0.0
    correct = 0;
    total = 0
    for good, bad in tqdm(data, desc="      [BLiMP]", leave=False, ncols=80):
        def score(s):
            inp = tokenizer(s, return_tensors="pt").to(model.device)
            lbl = inp.input_ids.clone()
            with torch.no_grad(): return -model(**inp, labels=lbl).loss.item()

        if score(good) > score(bad): correct += 1
        total += 1
    return (correct / (total + 1e-9)) * 100


def get_hidden_states(model, tokenizer, data, layer, limit=200):
    states = []
    model.eval()
    print(f"   [State Collection] Collecting L{layer} states for {len(data[:limit])} samples...")

    with torch.no_grad():
        for i, item in enumerate(tqdm(data[:limit], desc="Collecting")):
            # --- 1. Data cleaning ---
            try:
                if isinstance(item, tuple):
                    text = item[0]
                else:
                    text = item

                if not isinstance(text, str):
                    # Skip non-text data
                    continue

                prompt = f"<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\n{text}<|eot_id|>"
                inp = tokenizer(prompt, return_tensors="pt").to(model.device)

                # --- 2. Capture hidden states ---
                with nethook.Trace(model, f"model.layers.{layer}") as ret:
                    _ = model(**inp)

                    if ret.output is None:
                        print(f"❌ Error: nethook failed to capture layer {layer} output.")
                        continue

                    # Core fix: Intelligently determine the output type
                    output_obj = ret.output

                    # Case A: If it's a tuple (Standard HF), take the first element
                    if isinstance(output_obj, tuple):
                        output_tensor = output_obj[0]
                    else:
                        # Case B: If it's directly a tensor
                        output_tensor = output_obj

                    # Move to CPU for processing
                    output_tensor = output_tensor.detach().cpu()

                    # --- 3. Extract the Last Token ---
                    # output_tensor could be (1, Seq, Dim) or (Seq, Dim)

                    if output_tensor.ndim == 3:
                        # Shape: [Batch=1, Seq, Dim] -> take [0, -1, :]
                        hs = output_tensor[0, -1, :].numpy()
                    elif output_tensor.ndim == 2:
                        # Shape: [Seq, Dim] -> take [-1, :]
                        hs = output_tensor[-1, :].numpy()
                    else:
                        print(f"⚠️ Unexpected shape {output_tensor.shape} at index {i}")
                        continue

                    states.append(hs)

            except Exception as e:
                print(f"\n❌ Error processing sample {i}: {e}")
                # Continue instead of break to avoid interrupting the whole program
                continue

    return np.array(states)



# ==============================================================================
# MAIN
# ==============================================================================
def main():
    args = parse_args()
    print(f"Loading Model: {args.model_path}")
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    model = AutoModelForCausalLM.from_pretrained(args.model_path, device_map="auto", torch_dtype=torch.bfloat16)

    # 1. Load Datasets
    datasets = {}
    #if "mquake" in args.tasks:
        #d = args.mquake_dir if args.mquake_dir else os.path.join(args.data_root, "AKEW")
        #datasets['MQuAKE'] = load_mquake_data(d, args.probe_limit)
    if "gsm8k" in args.tasks:
        d = args.mquake_dir if args.mquake_dir else os.path.join(args.data_root, "AKEW")
        data = load_gsm8k_data(d, args.probe_limit)
        if not data: data = load_gsm8k_data(os.path.join(args.data_root, "gsm8k"), args.probe_limit)
        datasets['GSM8K'] = data
    if "math" in args.tasks:
        d = args.mquake_dir if args.mquake_dir else os.path.join(args.data_root, "AKEW")
        data = load_math_data(d, args.probe_limit)
        if not data: data = load_math_data(os.path.join(args.data_root, "math"), args.probe_limit)
        datasets['MATH'] = data
    #if "blimp" in args.tasks:
        #d = args.blimp_dir if args.blimp_dir else os.path.join(args.data_root, "blimp")
        #datasets['BLiMP'] = load_blimp_fixed(d)
    #datasets = {k: v for k, v in datasets.items() if v}

    # 2. Baselines
    print(f"\n[Calculating Baselines]")
    base_scores = {}
    for name, data in datasets.items():
        if name == 'BLiMP':
            s = eval_accuracy(model, tokenizer, data)
        else:
            s = eval_logprob(model, tokenizer, data)
        base_scores[name] = s
        print(f"   {name:<8}: {s:.4f}")

    # =========================================================================
    # MAIN EXPERIMENT LOOP
    # =========================================================================

    for seed in args.seeds:
        print(f"\n\n{'#' * 80}\n# STARTING SEED: {seed}\n{'#' * 80}")

        # -------------------------------------------------------------
        # Exp 1: GSM8K Sweep (MOM2=128 Fixed)
        # Variable: REMA Rank (k)
        # Fixed: MOM2 (always k=128)
        # Optimization: Only run REMA and Hybrid this round; run MOM2 once as baseline or check Exp 3
        # -------------------------------------------------------------
        if args.run_gsm8k_sweep:
            print(f"\n>>> [Seed {seed}] Task 1: GSM8K Rank Sweep (Methods: REMA, Hybrid)")
            for k in [1, 16, 64, 128]:
                # target_methods only contains methods affected by k
                run_experiment_step(model, tokenizer, datasets, base_scores, args,
                                    args.layer, 'gsm8k', k, 128, seed,
                                    target_methods=['Hybrid'], target_scales=[0.2])

                # -------------------------------------------------------------
        # Exp 2: MATH Sweep (MOM2=128 Fixed)
        # Optimization: Same as above, only run REMA and Hybrid
        # -------------------------------------------------------------
        if args.run_math_sweep:
            print(f"\n>>> [Seed {seed}] Task 2: MATH Rank Sweep (Methods: REMA, Hybrid)")
            for k in [1, 16, 64]:
                run_experiment_step(model, tokenizer, datasets, base_scores, args,
                                    args.layer, 'math', k, 128, seed,
                                    target_methods=['Hybrid'], target_scales=[0.2])

        # -------------------------------------------------------------
        # Exp 3: MOM2 Sweep (REMA Fixed)
        # Variable: MOM2 Rank (k)
        # Fixed: REMA
        # Optimization: Run the MOM2 family and Hybrid this round; REMA is fixed so no need to run it 8 times
        # -------------------------------------------------------------
        if args.run_mom2_sweep:
            print(f"\n>>> [Seed {seed}] Task 3: MOM2 Rank Sweep (Dual Configs)")

            # 1. Fix MATH=64, Sweep MOM2 (existing experiment)
            print(f"   >>> Sub-task 3.1: REMA=math (k=64)")
            for k in [1, 16, 64, 128]:
                run_experiment_step(model, tokenizer, datasets, base_scores, args,
                                    args.layer, 'gsm8k', 128, k, seed, target_methods=['Hybrid', 'MOM2_Self'], target_scales=[0.2])

            # 2. Fix GSM8K=64, Sweep MOM2 (new experiment!)
            # We choose k=64 as the fixed rank for GSM8K (it performed well and stably in Exp 1)
            print(f"   >>> Sub-task 3.2: REMA=gsm8k (k=64)")
            for k in [1, 16, 64]:
                run_experiment_step(model, tokenizer, datasets, base_scores, args,
                                    args.layer, 'math', 64, k, seed, target_methods=['Hybrid', 'MOM2_Self'], target_scales=[0.2])

        # -------------------------------------------------------------
        # Exp 4: Layer Sweep (L6, L16, L26)
        # Variable: Layer (when the layer changes, all matrices change)
        # Optimization: Must run everything
        # -------------------------------------------------------------
        if args.run_layer_sweep:
            print(f"\n>>> [Seed {seed}] Task 4: Layer Sweep (All Methods)")
            target_layers = [6, 16, 26]
            for layer_idx in target_layers:
                # Sub-task A: MATH Config
                run_experiment_step(model, tokenizer, datasets, base_scores, args,
                                    layer_idx, 'math', 64, 128, seed,
                                    target_methods=['MOM2_Self'], target_scales=[0.2])

                # Sub-task B: GSM8K Config
                run_experiment_step(model, tokenizer, datasets, base_scores, args,
                                    layer_idx, 'gsm8k', 128, 128, seed,
                                    target_methods=['MOM2_Self'], target_scales=[0.2])

        if args.run_debug:
            print(f"\n⚡⚡⚡ DEBUG MODE ACTIVATED: Printing Full Generations ⚡⚡⚡")

            # 1. Auto-detect model type and set Alpha (Scale)
            # Qwen -> 0.2, Llama -> 0.05
            model_id = args.model_path.lower()
            if "qwen" in model_id:
                target_scale = [0.2]
                print(f"🤖 Detected Qwen Model: Setting Alpha (Scale) = 0.2")
            else:
                target_scale = [0.05]
                print(f"🦙 Detected Llama/Other: Setting Alpha (Scale) = 0.05")

            # 2. Force the user-requested parameters
            debug_layer = 16
            debug_mom2_k = 128  # Fixed MOM2 rank
            debug_rema_k = 16  # Fixed REMA rank (Logic Sparsity)
            debug_seed = args.seeds[0] if args.seeds else 42

            print(f"⚙️  Config: Layer={debug_layer} | k_rema={debug_rema_k} | k_mom2={debug_mom2_k}")

            # 3. Run task A: GSM8K (Hybrid) + MOM2 Baseline
            print(f"\n>>> [DEBUG JOB 1] GSM8K Configuration & MOM2 Baseline")
            run_experiment_step(
                model, tokenizer, datasets, base_scores, args,
                layer=debug_layer,
                rema_type='math',  # Use GSM8K's REMA matrix
                rema_k=debug_rema_k,  # k=16
                mom2_k=debug_mom2_k,  # k=128
                seed=debug_seed,
                target_methods=['Hybrid', 'MOM2_Self'],
                target_scales=target_scale,
                verbose=True  # <--- Key: Enable verbose printing
            )

            # 4. Run task B: MATH (Hybrid)
            print(f"\n>>> [DEBUG JOB 2] MATH Configuration")
            run_experiment_step(
                model, tokenizer, datasets, base_scores, args,
                layer=debug_layer,
                rema_type='math',  # Use MATH's REMA matrix
                rema_k=debug_rema_k,  # k=16
                mom2_k=debug_mom2_k,  # k=128
                seed=debug_seed,
                target_methods=['Hybrid'],
                target_scales=target_scale,
                verbose=True  # <--- Key: Enable verbose printing
            )

            print("\n✅ Debug run completed. Exiting.")
            import sys
            sys.exit(0)

        if args.run_ablation:
            print(f"\n>>> [Seed {seed}] Task Ablation: RandCov vs RandVec")
            # Use the best configuration: L16, Scale 0.05, GSM-64, MOM2-128
            run_experiment_step(model, tokenizer, datasets, base_scores, args,
                                layer=16, rema_type='gsm8k', rema_k=16, mom2_k=128, seed=seed,
                                target_methods=['Hybrid', 'Ablation_RandCov', 'Ablation_RandVec'],
                                target_scales=[0.0, 0.05, 0.10, 0.15])

            run_experiment_step(model, tokenizer, datasets, base_scores, args,
                                layer=16, rema_type='math', rema_k=16, mom2_k=128, seed=seed,
                                target_methods=['Hybrid', 'Ablation_RandCov', 'Ablation_RandVec'],
                                target_scales=[0.0, 0.05, 0.10, 0.15])
            # Ablation is best viewed at a single Scale

            # -------------------------------------------------------------
            # Task: Main Figure Capture (controlled by --run_capture)
            # Only run once with Seed 42
            # -------------------------------------------------------------
        if seed == 42 and args.run_capture:
            # Ensure MQuAKE data exists (skip or do a dummy check if not loaded, to avoid crashes)
            if 'mquake' not in datasets:
                print("⚠️ Warning: MQuAKE dataset missing for capture task.")

            run_main_figure_capture(model, tokenizer, datasets, args, layer=16, seed=42)


    print("\n✅ All Experiments Complete.")


if __name__ == "__main__":
    main()