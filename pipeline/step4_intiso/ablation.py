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
# ⚙️ 配置与常量
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
# 📂 智能文件加载器
# ==============================================================================
def find_file(directory, pattern_keywords, extension, strict=True):
    if not os.path.exists(directory):
        if strict:
            print(f"   [Warning] Directory not found: {directory}")
            return None  # 降级为 Warning，不报错
        return None

    candidates = []
    for root, _, files in os.walk(directory):
        for file in files:
            if file.endswith(extension):
                if all(str(k).lower() in file.lower() for k in pattern_keywords):
                    candidates.append(os.path.join(root, file))

    if not candidates:
        if strict:
            # 降级为 Warning，防止整个程序崩溃
            print(f"   [Warning] No {extension} file found in {directory} matching {pattern_keywords}")
        return None

    candidates.sort(key=len)
    return candidates[0]


def run_debug_generation(model, tokenizer, layer, method, scale):
    """
    [Fixed] 执行 15-Shot 采样，自动适配 Llama-3 / Qwen 模板，精准截取生成内容
    """
    print(f"\n      🔎 [DEBUG SAMPLE] L{layer} | {method} | Scale={scale * 100:.0f}%")
    print(f"      {'=' * 50}")

    # 获取模型名称以选择模板
    model_name = tokenizer.name_or_path.lower() if tokenizer.name_or_path else ""

    for category, prompts in DEBUG_PROMPTS.items():
        print(f"      --- {category} ---")
        for i, prompt_text in enumerate(prompts):

            # 1. 🟢 智能构造 Prompt (适配 Qwen 和 Llama)
            if "qwen" in model_name:
                # Qwen ChatML 格式
                full_prompt = f"<|im_start|>user\n{prompt_text}<|im_end|>\n<|im_start|>assistant\n"
            elif "llama" in model_name or "llama3" in model_name:
                # Llama-3 格式
                full_prompt = f"<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\n{prompt_text}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
            else:
                # Fallback (Base Model style)
                full_prompt = f"Question: {prompt_text}\nAnswer:"

            # 2. 编码并记录输入长度
            inputs = tokenizer(full_prompt, return_tensors="pt").to(model.device)
            input_len = inputs.input_ids.shape[1]  # 关键：记录 Input 有多长

            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=512,
                    do_sample=False,  # 贪婪解码，保证复现性
                    temperature=0.0,
                    pad_token_id=tokenizer.eos_token_id
                )

            # 3. 🟢 精准截取：只解码新增的 tokens
            # outputs[0] 包含了 [Input_Ids + Generated_Ids]
            generated_ids = outputs[0][input_len:]
            generated = tokenizer.decode(generated_ids, skip_special_tokens=True).strip()

            # 压缩成一行显示 (去除换行符，保持 Log 整洁)
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
# 🧠 核心矩阵重构 (V9: Robust NPZ Support)
# ==============================================================================

import gc  # 确保导入 gc
from contextlib import contextmanager


@contextmanager
def temporary_parameter(model, param_name, new_value):
    """
    一个上下文管理器，用于临时修改模型参数，退出时自动恢复原始值。
    替代不存在的 nethook.set_parameter。
    """
    # 1. 获取参数对象
    # nethook.get_parameter 返回的是 nn.Parameter 对象
    param = nethook.get_parameter(model, param_name)

    # 2. 备份原始数据 (clone 以防引用被修改)
    old_data = param.data.clone()

    # 3. 应用新权重
    # 注意：必须修改 .data 属性，并确保类型/设备一致
    param.data = new_value.to(param.device, dtype=param.dtype)

    try:
        yield
    finally:
        # 4. 恢复原始数据 (无论中间是否报错)
        param.data = old_data
        # print(f"   🔄 Restored parameter: {param_name}")


def extract_top_k_components(data, k, device):

    tensor = None

    # 1. 优先提取标准 Key
    if isinstance(data, dict):
        priority_keys = ['projection_matrix', 'U', 'u', 'eigenvectors', 'eigvecs', 'mom2']
        for key in priority_keys:
            if key in data:
                tensor = data[key]
                break

        # 2. 兜底搜索：找符合模型维度的大矩阵
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

    #  3. 错误拦截
    if tensor is None:
        found_keys = list(data.keys()) if isinstance(data, dict) else "N/A"
        raise ValueError(
            f"\n❌ CRITICAL ERROR: 无法在文件中找到有效矩阵。\n"
            f"   文件包含 Keys: {found_keys}\n"
            f"   原因: 没有 Tensor 符合 Qwen (3584) 或 Llama (4096) 的维度。\n"
            f"   👉 请检查: 您是不是读取了 'spectral_data' (谱分析结果) 而不是矩阵文件？"
        )

    # 4. 准备处理
    del data
    gc.collect()
    torch.cuda.empty_cache()

    shape = tensor.shape
    dim_idx = -1

    # --- 维度判断 (兼容 Qwen) ---
    # Qwen Hidden (3584) or Llama Hidden (4096)
    if (3584 in shape or 4096 in shape) and (18944 not in shape and 14336 not in shape):
        target = 3584 if 3584 in shape else 4096
        dim_idx = 0 if shape[0] == target else 1

    # Qwen Inter (18944) or Llama Inter (14336)
    elif (18944 in shape or 14336 in shape) and (3584 not in shape and 4096 not in shape):
        target = 18944 if 18944 in shape else 14336
        dim_idx = 0 if shape[0] == target else 1

    elif shape[0] == shape[1]:  # 方阵
        dim_idx = -1

    # --- 提取 ---
    if dim_idx != -1:
        if dim_idx == 0:
            current_k = min(tensor.shape[1], k)
            result = tensor[:, :current_k]
        else:
            current_k = min(tensor.shape[0], k)
            result = tensor[:current_k, :].T
        return result.to(device, dtype=torch.float32)

    # --- 方阵分解 (Fallback) ---
    try:
        try:
            target_dtype = torch.bfloat16
            tensor_gpu = tensor.to(device, dtype=target_dtype)
        except:
            target_dtype = torch.float16
            tensor_gpu = tensor.to(device, dtype=target_dtype)

        tensor_gpu += torch.eye(shape[0], device=device, dtype=target_dtype) * 1e-4
        vals, vecs = torch.linalg.eigh(tensor_gpu)
        result = vecs[:, -k:]  # 取最大的 k 个特征向量

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

    # 🔍🔍🔍 DEBUG 打印：看看 Hybrid 到底在用什么原料 🔍🔍🔍
    print(f"\n   🔎 [DEBUG] Constructing Hybrid Matrix:")
    print(f"      Target Shape: {target_shape}")

    # 1. Load REMA (U)
    rema_path = find_file(args.rema_dir, ["rema", args.rema_type, f"L{args.layer}"], ".pt")
    print(f"      REMA Path (U): {rema_path}")

    if not rema_path:
        raise FileNotFoundError(f"REMA file not found in {args.rema_dir}")

    # 加载并检查 U 的维度
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

        # 🔍🔍🔍 DEBUG 打印：看看 REMA 到底读了哪个文件 🔍🔍🔍
        print(f"\n   🔎 [DEBUG] Loading {method} Matrix:")
        print(f"      Target Shape : {target_shape} (Model Layer)")
        print(f"      Source Path  : {source_path}")

        if not source_path:
            raise FileNotFoundError(f"Source file for {method} not found.")

        mat = reconstruct_matrix_from_source(source_path, k, device, target_shape)
        torch.save(mat, cache_path)
        return mat
    except Exception as e:
        # 打印更详细的错误堆栈
        import traceback
        print(f"   ❌ Error {method}: {e}")
        # traceback.print_exc() # 如果需要看代码行号，取消注释这行
        return torch.randn(target_shape, device=device)


import re
from collections import Counter
import numpy as np


# [保留你的 extract_answer 不变] ...

def calculate_fluency(text):
    """
    计算文本流利度 (基于加权 N-gram 熵)
    High Score = Rich/Diverse text
    Low Score (near 0) = Repetitive/Collapsed text
    """
    text = text.strip()
    if not text:
        return 0.0

    # 简单的分词逻辑
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

    # 你的加权逻辑
    weighted_e2 = e2 * (2 / 3)
    weighted_e3 = e3 * (4 / 3)

    score = np.mean([weighted_e2, weighted_e3])

    return score


def eval_generation_metrics(model, tokenizer, data, task_name, limit=500, batch_size=32):
    """
    [修复版] 修复 args 报错，增加 Debug 打印，智能匹配模板
    """
    correct = 0
    total_fluency = 0.0
    count = 0

    # 1. 截取数据
    test_data = data[:limit]

    # 2. Tokenizer 设置
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = 'left'  # Decoder 必须左填充

    # 3. 按 Batch 遍历
    for i in tqdm(range(0, len(test_data), batch_size), desc=f"   [Gen:{task_name}]", leave=False):
        batch = test_data[i: i + batch_size]

        # 3.1 批量构造 Prompts
        prompts = []
        golds = []

        # 🟢 修复点 1: 从 tokenizer 获取模型名称，不需要 args
        model_id = tokenizer.name_or_path.lower() if tokenizer.name_or_path else ""

        for q, a in batch:
            # 尝试使用标准 Chat 模板
            try:
                messages = [{"role": "user", "content": q}]
                full_prompt = tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True
                )
            except:
                # 🟢 修复点 2: 手动兜底策略 (Qwen 使用 ChatML)
                if "qwen" in model_id:
                    full_prompt = f"<|im_start|>user\n{q}<|im_end|>\n<|im_start|>assistant\n"
                elif "llama" in model_id:
                    full_prompt = f"<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\n{q}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
                else:
                    full_prompt = f"Question: {q}\nAnswer:"

            prompts.append(full_prompt)
            golds.append(str(a).lower().strip().replace(",", ""))

        # 3.2 批量编码
        inputs = tokenizer(prompts, return_tensors="pt", padding=True, truncation=True).to(model.device)
        input_len = inputs.input_ids.shape[1]

        eos_ids = [tokenizer.eos_token_id]  # 基础 EOS

        # 根据模型类型添加额外停止符
        if "qwen" in model_id:
            eos_ids.append(tokenizer.convert_tokens_to_ids("<|im_end|>"))
        elif "llama-3" in model_id or "llama3" in model_id:
            eos_ids.append(tokenizer.convert_tokens_to_ids("<|eot_id|>"))

        # 🧹 清洗列表：过滤掉 None 和非整数
        valid_eos_ids = [tid for tid in eos_ids if tid is not None and isinstance(tid, int)]

        generate_kwargs = {
            "max_new_tokens": 512,
            "do_sample": False,
            "temperature": 0.0,
            "pad_token_id": tokenizer.eos_token_id
        }
        if valid_eos_ids:
            generate_kwargs["eos_token_id"] = valid_eos_ids

        # 3.3 批量生成
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=512,
                do_sample=False,
                temperature=0.0,
                pad_token_id=tokenizer.eos_token_id,
                # 🟢 修复点 3: 显式指定 EOS token，防止 Qwen 停不下来
                eos_token_id=[tokenizer.eos_token_id, tokenizer.convert_tokens_to_ids("<|im_end|>")]
            )

        # 3.4 批量解码与评估
        for j, output in enumerate(outputs):
            gen_tokens = output[input_len:]
            gen_text = tokenizer.decode(gen_tokens, skip_special_tokens=True).strip()

            # 🔴 DEBUG 核心：打印前几个生成的文本，看看到底是个啥
            #if count < 3:
                #print(f"\n   🔎 [DEBUG Gen] Prompt Tail: ...{prompts[j][-50:].replace(chr(10), ' ')}")
                #print(f"   🔎 [DEBUG Gen] Output: {gen_text}")
                #print(f"   🔎 [DEBUG Gen] Gold:   {golds[j]}")

            # 1. Acc
            pred = extract_answer(gen_text, task_name)

            # MQuAKE 特殊处理
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
    """从生成文本中提取核心答案"""
    text = text.split("assistant")[-1].strip().lower() # 仅处理 assistant 部分
    if task_type.lower() == 'gsm8k':
        # 匹配最后的数字，支持逗号千分位
        match = re.findall(r"(\d+(?:,\d+)?(?:\.\d+)?)", text)
        if match:
            return match[-1].replace(",", "")
    elif task_type.lower() == 'math':
        # 优先寻找 \boxed{...}
        match = re.search(r"\\boxed\{(.*?)\}", text)
        if match: return match.group(1).strip()
        # 备选：找最后的数字
        nums = re.findall(r"(\d+(?:\.\d+)?)", text)
        return nums[-1] if nums else None
    elif task_type.lower() == 'mquake':
        return text # MQuAKE 通常直接做字符串包含检测
    return None


# ==============================================================================
# 🛠️ 新增工具函数
# ==============================================================================

def run_baseline_robustness(model, tokenizer, datasets, seeds=[42, 123, 999], limit=200):
    """
    测一组三个seed的原始模型对这200个问题的准确率
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
            # 使用 eval_generation_metrics 进行准确率和流利度评估
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
    收集数据跑 t-SNE 图的工具函数
    """
    os.makedirs(output_dir, exist_ok=True)
    print(f"\n🎨 Collecting t-SNE Data (L{layer})...")

    all_records = []

    for label, data in datasets.items():
        print(f"   -> Processing {label} ({len(data[:limit])} samples)...")

        # 清洗数据，提取文本
        clean_texts = []
        for item in data[:limit]:
            if isinstance(item, tuple):
                clean_texts.append(item[0])  # 取 Question 或 Sentence
            else:
                clean_texts.append(item)

        # 复用已有的 get_hidden_states 函数
        # 注意：get_hidden_states 返回的是 numpy array [N, Hidden_Dim]
        states = get_hidden_states(model, tokenizer, clean_texts, layer, limit=limit)

        # 封装数据
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
    封装单次实验逻辑
    新增参数 target_methods: list, 指定只跑哪些方法 (用于去重)
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
        # 🔍 V2: 增强版搜索，显式排除 spectral 分析文件
        if not os.path.exists(args.rema_dir): return None

        candidates = []
        for root, _, files in os.walk(args.rema_dir):
            for file in files:
                # 必须包含 rema, math/gsm8k, L16 这些关键词
                if file.endswith(".pt") and all(str(k) in file for k in ["rema", rema_type, f"L{layer}"]):

                    # 🛑 核心修复：如果您生成过谱分析图，文件夹里会有 spectral 文件
                    # 必须跳过它们，否则会读到只有 eigenvalues 的小文件 -> 导致报错
                    if "spectral" in file:
                        continue

                    candidates.append(os.path.join(root, file))

        if not candidates: return None
        # 按文件名长度排序，通常最短的是源文件
        candidates.sort(key=len)
        return candidates[0]

    def find_mom2_wrapper():
        path = find_file(args.mom2_dir, [f"layers.{layer}", "mom2"], ".npz", strict=False)
        if path: return path
        return find_file(args.mom2_dir, [f"L{layer}"], ".pt", strict=True)

    # --- 1. 按需加载矩阵 (节省 IO) ---
    # 如果 target_methods 指定了，就只加载需要的矩阵
    # 默认跑所有方法
    methods_to_run = target_methods if target_methods else ['Random', 'MOM2_Rand', 'MOM2_Self', 'REMA', 'Hybrid']

    # REMA / Hybrid 需要 REMA 矩阵
    if 'REMA' in methods_to_run or 'Hybrid' in methods_to_run:
        matrices['REMA'] = get_matrix_with_cache(args.cache_dir, "rema", rema_type, layer, rema_k,
                                                 model.device, target_shape, find_rema)

    # MOM2 / Hybrid 需要 MOM2 矩阵
    if any(m in methods_to_run for m in ['MOM2_Rand', 'MOM2_Self', 'Hybrid']):
        mom2_path = find_mom2_wrapper()
        if mom2_path and mom2_path.endswith(".npz"):
            d = np.load(mom2_path)
            data_mom2 = {k: torch.from_numpy(v) for k, v in d.items() if v.dtype.kind in {'f', 'i'}}
        else:
            data_mom2 = torch.load(mom2_path, map_location='cpu')  # 暂时加载到CPU

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

    # Hybrid 需要两者
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
        # 1. 加载真实的 REMA U (4096, k)
        rema_path = find_file(args.rema_dir, ["rema", rema_type, f"L{layer}"], ".pt")
        if rema_path:
            d_r = torch.load(rema_path, map_location=model.device)
            U_real = extract_top_k_components(d_r, rema_k, model.device)
            if U_real.shape[0] != target_shape[0]: U_real = U_real.T
            U_real = U_real[:, :rema_k]
        else:
            U_real = torch.randn(target_shape[0], rema_k, device=model.device)  # Fallback

        # 2. 加载真实的 MOM2 V (14336, k)
        mom2_path = find_mom2_wrapper()
        if mom2_path:
            if mom2_path.endswith(".npz"):
                d = np.load(mom2_path)
                d_m = {k: torch.from_numpy(v) for k, v in d.items() if v.dtype.kind in {'f', 'i'}}
            else:
                d_m = torch.load(mom2_path, map_location=model.device)
            V_real = extract_top_k_components(d_m, mom2_k, model.device)

            # 1. 严格转置逻辑：只有当第1维是目标维度时才转置
            if V_real.shape[0] != target_shape[1] and V_real.shape[1] == target_shape[1]:
                V_real = V_real.T

            # 2. 强制维度检查：防止 [128, 128] 这种错误形状混过去
            if V_real.shape[0] != target_shape[1]:
                print(
                    f"   ⚠️ Warning: V_real shape mismatch {V_real.shape}, expected dim0={target_shape[1]}. Replacing with Randn.")
                V_real = torch.randn(target_shape[1], mom2_k, device=model.device)

            V_real = V_real[:, :mom2_k]
        else:
            V_real = torch.randn(target_shape[1], mom2_k, device=model.device)  # Fallback

        # 确保秩一致
        rank = min(U_real.shape[1], V_real.shape[1])
        U_real = U_real[:, :rank]
        V_real = V_real[:, :rank]

        # --- 构造 Ablation 矩阵 ---

        # Case A: MOM2_Rand (Random Covariance) -> Real REMA @ Random V
        # 证明: 必须去除非随机的语法噪音 (Random V 代表随机噪音方向)
        if 'Ablation_RandCov' in methods_to_run:
            V_rand = torch.randn_like(V_real)
            # 正交化以模拟真实的基
            V_rand, _ = torch.linalg.qr(V_rand)
            matrices['Ablation_RandCov'] = U_real @ V_rand.T

        # Case B: Directionless REMA (Random Vector) -> Random U @ Real MOM2
        # 证明: 必须有明确的推理方向 (Random U 代表无方向)
        if 'Ablation_RandVec' in methods_to_run:
            U_rand = torch.randn_like(U_real)
            U_rand, _ = torch.linalg.qr(U_rand)
            matrices['Ablation_RandVec'] = U_rand @ V_real.T

    # Normalize
    for k, v in matrices.items(): matrices[k] = v / (v.norm() + 1e-8)

    # --- Eval Loop ---
    # Debug 模式只跑少量 Scale，正式实验跑全部
    #scales = [0.05]
    if target_scales:
        scales = target_scales
    else:
        scales = [0.05, 0.10, 0.15]
    weight_norm = param.norm().item()

    # 打印表头
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
                # 调用文件头部定义的 run_debug_generation
                # 它会打印 DEBUG_PROMPTS 定义的 5 个 GSM8K 和 5 个 MATH 例子
                run_debug_generation(model, tokenizer, layer, method, scale)

            results_str = []
            for name, data in datasets.items():
                if name == 'BLiMP':
                    res = f"{eval_accuracy(model, tokenizer, data):.2f}%"
                else:
                    # 🔥 修改点：使用 args.probe_limit 和 batch_size=16 (确保速度)
                    # 如果你显存大，可以把 batch_size 改成 32
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
    🎨 Main Figure Data Collector (Experimental Replica Version)
    严格复刻 run_experiment_step 的注入逻辑：
    1. Target: mlp.down_proj (4096 -> 14336)
    2. MOM2 Matrix: 强制识别为 14336 维 (V)
    3. REMA Matrix: 识别为 4096 维 (U)
    """
    print(f"\n{'=' * 80}")
    print(f"🎨 RUNNING MAIN FIGURE CAPTURE (Layer {layer}, Seed {seed})")
    print(f"{'=' * 80}")

    # ==========================================================================
    # 1. 准备 Prompts
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

    # 🎯 回归原始实验目标：down_proj
    layer_name = f"model.layers.{layer}.mlp.down_proj"

    try:
        W_old = nethook.get_parameter(model, f"{layer_name}.weight")
        print(f"   -> Target Weight ({layer_name}): {W_old.shape}")  # 应为 [4096, 14336]
        target_in_dim = W_old.shape[1]  # 14336
    except LookupError:
        print(f"❌ Layer {layer_name} not found.")
        return

    # ==========================================================================
    # (A) Load MOM2 (V) - 必须是 14336 维
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

        # 提取 k 个分量
        V_mom2 = extract_top_k_components(raw_mom2, mom2_k, model.device)

        # 🔥 关键：确保 V 是 [14336, k]
        # 如果是 [k, 14336]，转置
        # 如果是 [4096, k]，说明加载了错误的矩阵（Hidden），无法用于 down_proj
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
    # (B) Load REMA (U) - 必须是 4096 维
    # ==========================================================================
    print(f"   -> 🔍 Locating REMA Matrix (Type={rema_type}, k={rema_k})...")
    rema_path = find_file(args.rema_dir, ["rema", rema_type, f"L{layer}"], ".pt", strict=False)
    if not rema_path: return

    try:
        raw_rema = torch.load(rema_path, map_location=model.device)
        U_rema = extract_top_k_components(raw_rema, rema_k, model.device)
        # 确保 U 是 [4096, k]
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

    # 逻辑：Delta = scale * (W @ V) @ V.T
    # [4096, 14336] @ [14336, k] -> [4096, k]
    # [4096, k] @ [k, 14336] -> [4096, 14336]
    # 这样就能加上去了

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

    # 逻辑：Delta = scale * (U @ V.T)
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
# 📊 Eval Wrappers
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
            # --- 1. 数据清洗 ---
            try:
                if isinstance(item, tuple):
                    text = item[0]
                else:
                    text = item

                if not isinstance(text, str):
                    # 跳过非文本数据
                    continue

                prompt = f"<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\n{text}<|eot_id|>"
                inp = tokenizer(prompt, return_tensors="pt").to(model.device)

                # --- 2. 捕获 hidden states ---
                with nethook.Trace(model, f"model.layers.{layer}") as ret:
                    _ = model(**inp)

                    if ret.output is None:
                        print(f"❌ Error: nethook failed to capture layer {layer} output.")
                        continue

                    # 🔥 修复核心：智能判断 output 类型
                    output_obj = ret.output

                    # 情况 A: 如果是元组 (Standard HF)，取第一个元素
                    if isinstance(output_obj, tuple):
                        output_tensor = output_obj[0]
                    else:
                        # 情况 B: 如果直接是张量
                        output_tensor = output_obj

                    # 转 CPU 以便处理
                    output_tensor = output_tensor.detach().cpu()

                    # --- 3. 提取 Last Token ---
                    # 现在的 output_tensor 可能是 (1, Seq, Dim) 也可能是 (Seq, Dim)

                    if output_tensor.ndim == 3:
                        # 形状: [Batch=1, Seq, Dim] -> 取 [0, -1, :]
                        hs = output_tensor[0, -1, :].numpy()
                    elif output_tensor.ndim == 2:
                        # 形状: [Seq, Dim] -> 取 [-1, :]
                        hs = output_tensor[-1, :].numpy()
                    else:
                        print(f"⚠️ Unexpected shape {output_tensor.shape} at index {i}")
                        continue

                    states.append(hs)

            except Exception as e:
                print(f"\n❌ Error processing sample {i}: {e}")
                # 为了不中断整个程序，这里 continue 而不是 break
                continue

    return np.array(states)



# ==============================================================================
# 🚀 MAIN
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
    # 🔄 MAIN EXPERIMENT LOOP
    # =========================================================================

    for seed in args.seeds:
        print(f"\n\n{'#' * 80}\n# STARTING SEED: {seed}\n{'#' * 80}")

        # -------------------------------------------------------------
        # Exp 1: GSM8K Sweep (MOM2=128 Fixed)
        # 变化量: REMA Rank (k)
        # 固定量: MOM2 (一直也是 k=128)
        # 优化策略: 这一轮只跑 REMA 和 Hybrid，MOM2 跑一次基准或者看 Exp3 即可
        # -------------------------------------------------------------
        if args.run_gsm8k_sweep:
            print(f"\n>>> [Seed {seed}] Task 1: GSM8K Rank Sweep (Methods: REMA, Hybrid)")
            for k in [1, 16, 64, 128]:
                # target_methods 只包含受 k 影响的方法
                run_experiment_step(model, tokenizer, datasets, base_scores, args,
                                    args.layer, 'gsm8k', k, 128, seed,
                                    target_methods=['Hybrid'], target_scales=[0.2])

                # -------------------------------------------------------------
        # Exp 2: MATH Sweep (MOM2=128 Fixed)
        # 优化策略: 同上，只跑 REMA 和 Hybrid
        # -------------------------------------------------------------
        if args.run_math_sweep:
            print(f"\n>>> [Seed {seed}] Task 2: MATH Rank Sweep (Methods: REMA, Hybrid)")
            for k in [1, 16, 64]:
                run_experiment_step(model, tokenizer, datasets, base_scores, args,
                                    args.layer, 'math', k, 128, seed,
                                    target_methods=['Hybrid'], target_scales=[0.2])

        # -------------------------------------------------------------
        # Exp 3: MOM2 Sweep (REMA Fixed)
        # 变化量: MOM2 Rank (k)
        # 固定量: REMA
        # 优化策略: 这一轮跑 MOM2系列 和 Hybrid，REMA 是固定的没必要跑 8 遍
        # -------------------------------------------------------------
        if args.run_mom2_sweep:
            print(f"\n>>> [Seed {seed}] Task 3: MOM2 Rank Sweep (Dual Configs)")

            # 1. Fix MATH=64, Sweep MOM2 (现有实验)
            print(f"   >>> Sub-task 3.1: REMA=math (k=64)")
            for k in [1, 16, 64, 128]:
                run_experiment_step(model, tokenizer, datasets, base_scores, args,
                                    args.layer, 'gsm8k', 128, k, seed, target_methods=['Hybrid', 'MOM2_Self'], target_scales=[0.2])

            # 2. Fix GSM8K=64, Sweep MOM2 (新增实验!)
            # 我们选择 k=64 作为 GSM8K 的固定秩 (因为它在 Exp 1 中表现优异且稳定)
            print(f"   >>> Sub-task 3.2: REMA=gsm8k (k=64)")
            for k in [1, 16, 64]:
                run_experiment_step(model, tokenizer, datasets, base_scores, args,
                                    args.layer, 'math', 64, k, seed, target_methods=['Hybrid', 'MOM2_Self'], target_scales=[0.2])

        # -------------------------------------------------------------
        # Exp 4: Layer Sweep (L6, L16, L26)
        # 变化量: Layer (层变了，所有矩阵都变了)
        # 优化策略: 必须全跑
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

            # 1. 自动判断模型类型并设置 Alpha (Scale)
            # Qwen -> 0.2, Llama -> 0.05
            model_id = args.model_path.lower()
            if "qwen" in model_id:
                target_scale = [0.2]
                print(f"🤖 Detected Qwen Model: Setting Alpha (Scale) = 0.2")
            else:
                target_scale = [0.05]
                print(f"🦙 Detected Llama/Other: Setting Alpha (Scale) = 0.05")

            # 2. 强制设定用户要求的参数
            debug_layer = 16
            debug_mom2_k = 128  # 固定 MOM2 秩
            debug_rema_k = 16  # 固定 REMA 秩 (Logic Sparsity)
            debug_seed = args.seeds[0] if args.seeds else 42

            print(f"⚙️  Config: Layer={debug_layer} | k_rema={debug_rema_k} | k_mom2={debug_mom2_k}")

            # 3. 执行任务 A: GSM8K (Hybrid) + MOM2 Baseline
            print(f"\n>>> [DEBUG JOB 1] GSM8K Configuration & MOM2 Baseline")
            run_experiment_step(
                model, tokenizer, datasets, base_scores, args,
                layer=debug_layer,
                rema_type='math',  # 指定使用 GSM8K 的 REMA 矩阵
                rema_k=debug_rema_k,  # k=16
                mom2_k=debug_mom2_k,  # k=128
                seed=debug_seed,
                target_methods=['Hybrid', 'MOM2_Self'],
                target_scales=target_scale,
                verbose=True  # <--- 🔥 关键：开启详细打印
            )

            # 4. 执行任务 B: MATH (Hybrid)
            print(f"\n>>> [DEBUG JOB 2] MATH Configuration")
            run_experiment_step(
                model, tokenizer, datasets, base_scores, args,
                layer=debug_layer,
                rema_type='math',  # 指定使用 MATH 的 REMA 矩阵
                rema_k=debug_rema_k,  # k=16
                mom2_k=debug_mom2_k,  # k=128
                seed=debug_seed,
                target_methods=['Hybrid'],
                target_scales=target_scale,
                verbose=True  # <--- 🔥 关键：开启详细打印
            )

            print("\n✅ Debug run completed. Exiting.")
            import sys
            sys.exit(0)

        if args.run_ablation:
            print(f"\n>>> [Seed {seed}] Task Ablation: RandCov vs RandVec")
            # 调用最佳配置: L16, Scale 0.05, GSM-64, MOM2-128
            run_experiment_step(model, tokenizer, datasets, base_scores, args,
                                layer=16, rema_type='gsm8k', rema_k=16, mom2_k=128, seed=seed,
                                target_methods=['Hybrid', 'Ablation_RandCov', 'Ablation_RandVec'],
                                target_scales=[0.0, 0.05, 0.10, 0.15])

            run_experiment_step(model, tokenizer, datasets, base_scores, args,
                                layer=16, rema_type='math', rema_k=16, mom2_k=128, seed=seed,
                                target_methods=['Hybrid', 'Ablation_RandCov', 'Ablation_RandVec'],
                                target_scales=[0.0, 0.05, 0.10, 0.15])
            # Ablation 建议只看一个 Scale

            # -------------------------------------------------------------
            # 🎨 Task: Main Figure Capture (受 --run_capture 控制)
            # 只在 Seed 42 跑一次
            # -------------------------------------------------------------
        if seed == 42 and args.run_capture:
            # 确保有 MQuAKE 数据 (如果没有加载，为了不报错可以跳过或做个假数据检查)
            if 'mquake' not in datasets:
                print("⚠️ Warning: MQuAKE dataset missing for capture task.")

            run_main_figure_capture(model, tokenizer, datasets, args, layer=16, seed=42)


    print("\n✅ All Experiments Complete.")


if __name__ == "__main__":
    main()