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
    # 注意：这里传入的是包含 .pt 文件的文件夹路径
    parser.add_argument("--matrix_dir", type=str, required=True, help="Directory containing REMA/Trim vectors")
    # 这里传入 mquake.json 的路径
    parser.add_argument("--dataset_path", type=str, required=True, help="Path to mquake.json")
    parser.add_argument("--layers", type=int, nargs="+", default=[12, 13, 14, 15, 16])
    parser.add_argument("--probe_limit", type=int, default=100, help="Number of multi-hop cases to test")
    return parser.parse_args()


def load_mquake_data(path, limit=50):
    """
    针对 MQuAKE 结构加载：(Multi-hop Question, Counterfactual Answer)
    """
    print(f"Loading MQuAKE data from: {path}")
    if not os.path.exists(path):
        raise FileNotFoundError(f"Dataset not found: {path}")

    qa_pairs = []
    with open(path, 'r') as f:
        # MQuAKE 通常是一个大的 List
        try:
            data = json.load(f)
        except json.JSONDecodeError:
            # 兼容 jsonl 格式
            f.seek(0)
            data = [json.loads(line) for line in f]

    for item in data:
        if len(qa_pairs) >= limit: break

        # 1. 获取多跳问题 (Multi-hop Questions)
        # MQuAKE 的 questions 字段是一个列表，包含同一个意思的不同问法
        # 我们取第一个即可
        q_list = item.get('questions', [])
        if not q_list: continue
        question = q_list[0]

        # 2. 获取多跳推理的目标答案 (New Answer / Counterfactual Answer)
        # 我们想测的是：REMA 信号是否触动了模型进行“反事实推理”的回路
        # 如果没有 new_answer (非编辑样本)，则回退到 answer
        target_answer = item.get('new_answer', item.get('answer'))

        if question and target_answer:
            qa_pairs.append((question, target_answer))

    print(f"Loaded {len(qa_pairs)} MQuAKE reasoning pairs.")
    # 打印一个样本示例，确认加载正确
    if qa_pairs:
        print(f"Sample Q: {qa_pairs[0][0]}")
        print(f"Sample A: {qa_pairs[0][1]}")

    return qa_pairs


def compute_masked_loss(model, tokenizer, question, answer):
    """
    构造 Prompt 并计算 Answer 部分的 Loss
    Prompt: <User> Question <Assistant> Answer
    """
    # Llama-3 模板
    prompt_template = "<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\n{}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
    prompt = prompt_template.format(question)
    full_text = prompt + answer

    # 编码
    enc_prompt = tokenizer(prompt, return_tensors="pt", add_special_tokens=False)
    enc_full = tokenizer(full_text, return_tensors="pt", add_special_tokens=False)

    input_ids = enc_full.input_ids.to(model.device)
    labels = input_ids.clone()

    # Mask 掉 Prompt 部分 (设为 -100)
    # 注意：要计算一下 prompt 的长度
    prompt_len = enc_prompt.input_ids.shape[1]
    labels[:, :prompt_len] = -100

    with torch.no_grad():
        outputs = model(input_ids=input_ids, labels=labels)
        return outputs.loss.item()


def measure_ablation_sensitivity(model, layer_idx, update_matrix, tokenizer, qa_pairs):
    """
    对比测试：REMA 信号 vs. 随机同能量噪声
    + [新增] 实时生成探针：观察前几个样本的实际输出
    """
    layer_name = f"model.layers.{layer_idx}.mlp.down_proj"
    weights = nethook.get_parameter(model, f"{layer_name}.weight")

    # --- 1. 准备 REMA 向量 (Signal) ---
    d_signal = update_matrix.to(weights.device).to(weights.dtype)
    signal_norm = d_signal.norm().item()
    if signal_norm == 0: return 0, 0

    # --- 2. 准备 Random 向量 (Noise) ---
    # 关键：保持能量 (Norm) 一致，方向随机
    torch.manual_seed(42 + layer_idx)  # 不同的层用不同的随机种子
    d_noise = torch.randn_like(d_signal)
    d_noise = d_noise / (d_noise.norm() + 1e-8) * signal_norm

    # --- 3. 设定微扰强度 (epsilon) ---
    # 扰动权重 Norm 的 1%
    epsilon = weights.norm().item() * 0.01

    # 归一化扰动向量
    pert_signal = (d_signal / (signal_norm + 1e-8)) * epsilon
    pert_noise = (d_noise / (d_noise.norm() + 1e-8)) * epsilon

    total_delta_signal = 0
    total_delta_noise = 0
    count = 0

    # Prompt 模板 (用于生成时的格式化)
    prompt_tpl = "<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\n{}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"

    print(f"  > Probing {len(qa_pairs)} MQuAKE samples...")

    for q, a in qa_pairs:
        try:
            # 0. Base Loss
            l0 = compute_masked_loss(model, tokenizer, q, a)

            # =========================================================
            # A. Signal Sensitivity (REMA)
            # =========================================================
            weights.data += pert_signal  # <--- 注入 REMA 信号

            # 计算 Loss
            l_signal = compute_masked_loss(model, tokenizer, q, a)

            # >>> [插入点] Case Study 生成测试 (只看前 3 个样本) <<<
            if count < 5:
                print(f"\n🔍 [Case Study L{layer_idx} | Signal] Q: {q}")
                # 构造标准对话输入
                prompt = prompt_tpl.format(q)
                inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

                # 生成 (Greedy Search, Max 30 tokens)
                with torch.no_grad():
                    gen_ids = model.generate(
                        **inputs,
                        max_new_tokens=30,
                        do_sample=False,
                        pad_token_id=tokenizer.eos_token_id
                    )

                # 解码 (只截取新生成的部分)
                input_len = inputs.input_ids.shape[1]
                gen_text = tokenizer.decode(gen_ids[0][input_len:], skip_special_tokens=True)

                # [修改点]：先处理字符串，避免 f-string 中出现反斜杠
                clean_output = gen_text.strip().replace('\n', ' ')

                print(f"   [Model Output]: {clean_output}")
                print(f"   [Target Answer]: {a}")

            weights.data -= pert_signal  # <--- 还原权重 (Restore)

            # =========================================================
            # B. Noise Sensitivity (Random)
            # =========================================================
            weights.data += pert_noise
            l_noise = compute_masked_loss(model, tokenizer, q, a)
            weights.data -= pert_noise  # Restore

            # 累加绝对变化量
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
    """加载并重建矩阵，兼容不同存储格式"""
    data = torch.load(pt_path, map_location=device)

    # Case A: SVD 格式 (U, S, V)
    if "u" in data and "s" in data and "v" in data:
        u = data["u"].to(dtype=torch.float16)
        s = data["s"].to(dtype=torch.float16)
        v = data["v"].to(dtype=torch.float16)
        # REMA 或是 投影矩阵，或是更新量，这里假设是 Update Delta
        return (u * s) @ v.T

    # Case B: 直接存储的矩阵
    elif "update_matrix" in data:
        return data["update_matrix"].to(dtype=torch.float16)

    # Case C: 只有 V (Projection Matrix)
    # 如果你存的是投影矩阵 P，你需要决定怎么把它转成扰动
    # 这里假设我们测试的是 Update Matrix
    # 如果文件里没有 Update Matrix，可能需要你手动指定逻辑
    else:
        # 尝试返回第一个可能是 Tensor 的值
        for k, v in data.items():
            if isinstance(v, torch.Tensor):
                return v.to(dtype=torch.float16)
    raise ValueError(f"Unknown matrix format in {pt_path}")


def main():
    args = parse_args()

    # 加载模型
    print(f"Loading Model: {args.model_path}")
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        device_map="auto",
        torch_dtype=torch.float16
    )

    # 加载 MQuAKE 数据
    qa_pairs = load_mquake_data(args.dataset_path, limit=args.probe_limit)

    print("\n" + "=" * 80)
    print(f"⚔️  REMA Logic Verification: Multi-hop Reasoning Sensitivity  ⚔️")
    print(f"Target: Does REMA affect the 'Counterfactual Answer' more than Random Noise?")
    print("=" * 80)
    print(f"{'Layer':<6} | {'Signal Sens.':<12} | {'Noise Sens.':<12} | {'Ratio (S/N)':<15} | {'Verdict'}")
    print("-" * 75)

    for layer in args.layers:
        # 寻找对应的矩阵文件 (根据你的命名习惯调整通配符)
        # 假设文件名类似: factors_trim_L12.pt 或 rema_update_L12.pt
        pattern = os.path.join(args.matrix_dir, f"*L{layer}*.pt")
        candidates = glob.glob(pattern)

        # 简单过滤：优先找带 'trim' 或 'rema' 的
        files = [f for f in candidates if 'trim' in f or 'rema' in f]
        if not files and candidates: files = candidates  # 没找到就用所有候选

        if not files:
            print(f"L{layer:<5} | No matrix file found.")
            continue

        target_file = files[0]  # 取第一个

        try:
            # 重建矩阵
            update_mat = reconstruct_matrix(target_file, device=model.device)

            # 执行测试
            sens_signal, sens_noise = measure_ablation_sensitivity(
                model, layer, update_mat, tokenizer, qa_pairs
            )

            # 计算信噪比
            ratio = sens_signal / (sens_noise + 1e-9)

            # 简单的判定
            verdict = "✅ VALID" if ratio > 1.2 else ("⚠️ WEAK" if ratio > 1.0 else "❌ NOISE")

            print(f"L{layer:<5} | {sens_signal:.4e}   | {sens_noise:.4e}   | {ratio:.2f}x            | {verdict}")

        except Exception as e:
            print(f"L{layer:<5} | Error: {e}")
            import traceback
            traceback.print_exc()


if __name__ == "__main__":
    main()