import sys, os
_PROJ_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _PROJ_ROOT not in sys.path:
    sys.path.insert(0, _PROJ_ROOT)

import os
import torch
import numpy as np
import argparse
from transformers import AutoModelForCausalLM, AutoTokenizer

# 引入项目中的工具函数 (根据你的项目结构调整 import)
# 假设这些在你的 util 和 rome 文件夹里
try:
    from utils import nethook
    # 你原来的代码里有 load_blimp_local 等函数，这里需要把它们加上
    # 为了保证能跑，请确保这里包含了 load_blimp_local 和 extract_syntax_activations 的定义
    # 或者 import 进来
    from pipeline.step3_verification.spectrum_blimp_mom2 import load_blimp_local, extract_syntax_activations, BLIMP_PARADIGMS
except ImportError:
    # 如果 import 失败，请把那两个辅助函数直接复制粘贴到这个文件里
    print(
        "Warning: Could not import helper functions. Please ensure analyze_spectrum.py is accessible or copy the helper functions here.")
    pass


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, required=True)
    parser.add_argument("--layer", type=int, required=True)
    parser.add_argument("--rema_path", type=str, required=True, help="Path to the REMA matrix .pt file")
    parser.add_argument("--blimp_root", type=str, default="/data/users/yanrongen/ROME-MEMIT/data/blimp")
    parser.add_argument("--output_dir", type=str, default="analysis_results")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # 1. Load Model
    print(f"Loading model: {args.model_name}...")
    model = AutoModelForCausalLM.from_pretrained(args.model_name, torch_dtype=torch.float16, device_map="auto")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    tokenizer.pad_token = tokenizer.eos_token

    # 2. Load REMA Matrix (替代原本的 Eigen decomposition)
    print(f"Loading REMA matrix from {args.rema_path}...")
    rema_data = torch.load(args.rema_path, map_location="cpu")

    # [关键] 直接提取 V 矩阵作为投影基
    if 'projection_matrix' in rema_data:
        eigvecs = rema_data['projection_matrix'].float().to(device)  # [Hidden, k]
    else:
        raise ValueError(f"Could not find 'projection_matrix' in {args.rema_path}")

    print(f"REMA V (Eigvecs) shape: {eigvecs.shape}")

    # [关键] 提取 REMA 的特征值 (用于画红线)
    if 'S' in rema_data:
        eigvals = (rema_data['S'].float() ** 2).to('cpu')
    elif 'eigenvalues' in rema_data:
        eigvals = rema_data['eigenvalues'].float().to('cpu')
    else:
        print("Warning: No eigenvalues found, creating dummy.")
        eigvals = torch.zeros(eigvecs.shape[1])

    # 3. Load BLiMP & Extract Activations
    print("Loading BLiMP dataset...")
    blimp_sentences = load_blimp_local(
        args.blimp_root,
        BLIMP_PARADIGMS,
        max_samples_per_paradigm=300,
    )

    layer_name = f"model.layers.{args.layer}"
    print(f"Extracting syntax activations from {layer_name}...")

    # 确保 extract_syntax_activations 可用
    syntax_matrix = extract_syntax_activations(
        model,
        tokenizer,
        layer_name,
        blimp_sentences,
        device,
    ).to(device)

    # 4. Compute Grammar Energy (Projection)
    print("Computing spectral projections (Chunked)...")
    num_samples = syntax_matrix.shape[0]
    chunk_size = 100
    projections_list = []
    eigvecs = eigvecs.float()

    with torch.no_grad():
        for i in range(0, num_samples, chunk_size):
            chunk = syntax_matrix[i: i + chunk_size].float()

            # 投影: [Batch, Hidden] @ [Hidden, k] -> [Batch, k]
            proj_chunk = chunk @ eigvecs

            energy_chunk = proj_chunk ** 2

            projections_list.append(energy_chunk.cpu())
            torch.cuda.empty_cache()

    projections = torch.cat(projections_list, dim=0)
    avg_grammar_energy = projections.mean(dim=0)  # [k]

    # 5. Save
    os.makedirs(args.output_dir, exist_ok=True)
    save_path = os.path.join(
        args.output_dir,
        f"layer_rema_{args.layer}_spectral_data.pt"
    )

    torch.save(
        {
            "layer": args.layer,
            "eigenvalues": eigvals,
            "grammar_projection": avg_grammar_energy,
            "num_blimp_sentences": len(blimp_sentences),
        },
        save_path,
    )
    print(f"✅ Analysis complete. Saved to {save_path}")


if __name__ == "__main__":
    main()