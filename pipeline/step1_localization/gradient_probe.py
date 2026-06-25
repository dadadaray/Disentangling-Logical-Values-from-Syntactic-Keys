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
# 1. 工具函数
# -------------------------------------------------
def find_subsequence(sequence, subseq):
    """在长序列中寻找子序列的位置"""
    L, l = len(sequence), len(subseq)
    for i in range(L - l + 1):
        if sequence[i:i + l] == subseq:
            return list(range(i, i + l))
    return None


def parse_mquake_item(item):
    """
    解析 MQUAKE 数据，同时提取 Single-hop 和 Multi-hop 问题
    """
    results = []

    # 基本校验
    if "questions" not in item or "requested_rewrite" not in item:
        return []
    if len(item["requested_rewrite"]) == 0:
        return []

    # 提取 Subject (通常在 requested_rewrite 的第一个里)
    subject = item["requested_rewrite"][0].get("subject")

    # A. 提取 Multi-hop (多跳推理)
    # Target 是 MQUAKE 的最终答案 (new_answer)
    multi_hop_target = item.get("new_answer")
    for q in item["questions"]:
        if q and multi_hop_target:
            results.append({
                "type": "multi_hop",
                "prompt": q,
                "subject": subject,
                "target": multi_hop_target
            })

    # B. 提取 Single-hop (单跳记忆)
    # Target 是单跳问题的答案 (通常是 edited fact 的 object)
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
# 2. 核心 Hook：噪音注入 & 状态修复
# -------------------------------------------------
@contextmanager
def add_noise_to_subject(model, token_positions, noise_std=0.1):
    """
    [Context Manager] 在 Embedding 层给指定位置添加高斯噪音
    用于制造 "Corrupt" 状态，但不改变 Sequence Length
    """
    hooks = []

    def noise_hook_fn(module, inp, out):
        # out 通常是 (batch, seq, dim) 的 Tensor
        # 有些模型输出是 tuple，取第0个
        if isinstance(out, tuple):
            tensor = out[0]
        else:
            tensor = out

        # 生成噪音 (保持设备和类型一致)
        # 注意：这里我们给 batch 里所有样本都加噪音，假设 batch_size=1
        noise = torch.randn_like(tensor[:, token_positions, :]) * noise_std

        # ⚠️ 必须 clone，否则原地修改可能会报错或影响梯度（虽然这里是 no_grad）
        cloned = tensor.clone()
        cloned[:, token_positions, :] += noise

        if isinstance(out, tuple):
            return (cloned,) + out[1:]
        return cloned

    # 自动寻找 Embedding 层
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
    [Context Manager] 在指定层强行将 Hidden States 替换回 Clean Run 的状态
    """
    hooks = []

    def hook_fn(module, inp, out):
        if isinstance(out, tuple):
            target = out[0]
        else:
            target = out

        # 遍历需要修复的位置 (Subject positions)
        for pos in token_positions:
            if pos < target.shape[1]:
                # 强行覆盖为 Clean Hidden
                target[:, pos, :] = clean_hidden[:, pos, :]
        return out

    # 适配不同模型架构定位 Layer
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
# 3. 主 Probe 循环 (Causal Tracing)
# -------------------------------------------------
@torch.no_grad()
def run_hop_causal_probe(model, tokenizer, dataset, num_samples=50):
    device = next(model.parameters()).device
    # 获取层数
    if hasattr(model.config, "num_hidden_layers"):
        num_layers = model.config.num_hidden_layers
    else:
        num_layers = model.config.n_layer

    results = []

    # 动态计算噪音强度：参考 ROME 论文，设为 Embedding 标准差的 3 倍
    embeddings = model.get_input_embeddings().weight
    noise_std = (embeddings.std() * 3).item()
    print(f"[*] Calculated noise_std: {noise_std:.4f}")

    # 限制样本数
    process_data = dataset[:num_samples] if num_samples > 0 else dataset

    for item in tqdm(process_data, desc="Processing Samples"):
        # 1. 解析样本 (包含 Single 和 Multi hop)
        parsed_items = parse_mquake_item(item)

        for p_item in parsed_items:
            prompt = p_item["prompt"]
            subject = p_item["subject"]
            target = p_item["target"]
            hop_type = p_item["type"]  # "single_hop" or "multi_hop"

            # 2. Tokenize & 定位
            # Llama-3 等模型可能会加特殊 Token，使用 return_offsets_mapping 或手动查找
            inputs = tokenizer(prompt, return_tensors="pt").to(device)
            input_ids_list = inputs.input_ids[0].tolist()

            # 寻找 Subject Token 位置
            subject_ids = tokenizer(subject, add_special_tokens=False).input_ids
            # 简单的查找逻辑
            pos = find_subsequence(input_ids_list, subject_ids)

            # 如果没找到 Subject (可能是分词导致的前导空格问题)，尝试加个空格再找
            if not pos:
                subject_ids = tokenizer(" " + subject, add_special_tokens=False).input_ids
                pos = find_subsequence(input_ids_list, subject_ids)

            if not pos:
                # 实在找不到就跳过，避免报错
                continue

            # 获取目标答案的 Token ID (用于看 Logit 变化)
            # 取答案的第一个 Token 作为观察点
            target_ids = tokenizer(target, add_special_tokens=False).input_ids
            if len(target_ids) == 0: continue
            answer_token_id = target_ids[0]

            # -------- Step A: Clean Run (基准线) --------
            # 跑一次干净的，保存所有层的 Hidden States
            clean_outputs = model(**inputs, output_hidden_states=True)
            clean_hidden_cache = clean_outputs.hidden_states
            # clean_hidden_cache[i] 是第 i 层的输出 (Llama通常有 33 个元素: 1个Embed + 32个Layer)

            # 记录基准概率 (Clean Score)
            clean_logit = clean_outputs.logits[0, -1, answer_token_id].item()

            # -------- Step B: Causal Tracing (Patching) --------
            # 遍历每一层，尝试 "修复" 记忆
            for layer_idx in range(num_layers):
                # 1. 制造破坏：在 Embedding 层加噪音 (Corrupt)
                with add_noise_to_subject(model, pos, noise_std=noise_std):
                    # 2. 尝试修复：把 layer_idx 的 Subject 状态替换回 Clean 状态
                    # 注意索引：model.layers[i] 的输出对应 hidden_states[i+1]
                    with patch_layer_hidden(
                            model, layer_idx, pos, clean_hidden_cache[layer_idx + 1]
                    ):
                        # 3. 前向传播并观测结果
                        patch_out = model(**inputs)
                        patch_logit = patch_out.logits[0, -1, answer_token_id].item()

                # 4. 记录数据
                results.append({
                    "case_id": item.get("case_id"),
                    "hop_type": hop_type,  # 关键区分点: single vs multi
                    "layer": layer_idx,
                    "clean_logit": clean_logit,
                    "patched_logit": patch_logit,
                    # 可以计算恢复分数: restored = patched - corrupted (需另测 corrupt)
                    # 或者直接存 patch_logit 后期画图分析
                })

            del clean_outputs, clean_hidden_cache
            torch.cuda.empty_cache()

    return results


# -------------------------------------------------
# 4. 程序入口
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
