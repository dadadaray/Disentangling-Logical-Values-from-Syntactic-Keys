import sys, os
_PROJ_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _PROJ_ROOT not in sys.path:
    sys.path.insert(0, _PROJ_ROOT)

import torch
import numpy as np
import os
import argparse
from transformers import AutoModelForCausalLM, AutoTokenizer
from utils.layer_stats_mom2 import layer_stats
import torch.nn.functional as F


# --- Helper Functions ---
def safe_load_matrix(path, device, top_k=None):
    if not os.path.exists(path):
        if os.path.exists(path + ".pt"):
            path += ".pt"
        else:
            return None
    try:
        data = torch.load(path, map_location=device)
        U = None
        if isinstance(data, torch.Tensor):
            U = data
        elif isinstance(data, dict):
            for k in ['U', 'V', 'matrix', 'components', 'projection_matrix']:
                if k in data:
                    U = data[k];
                    break
            if U is None: U = list(data.values())[0]
        if U is None: return None
        if U.shape[0] < U.shape[1]: U = U.T
        if top_k is not None:
            k_eff = min(U.shape[1], top_k)
            U = U[:, :k_eff]
        U = torch.nn.functional.normalize(U, p=2, dim=0)
        return U.float()
    except Exception as e:
        print(f"Error loading {path}: {e}")
        return None


def compute_metrics(U_base, U_sweep):
    # Interaction matrix: [K_sweep, K_base]
    interaction = torch.matmul(U_sweep.T, U_base)

    # Metric 1: Max Cosine Similarity (the strictest metric)
    max_overlap = torch.max(torch.abs(interaction)).item()

    # Metric 2: Avg Leakage
    projected_energy = torch.norm(interaction) ** 2
    total_energy = min(U_base.shape[1], U_sweep.shape[1])
    avg_overlap = projected_energy / (total_energy + 1e-8)

    return max_overlap, avg_overlap.item()


def project_rema_to_mlp(model, layer_idx, U_rema):
    mlp = model.model.layers[layer_idx].mlp
    device = mlp.gate_proj.weight.device
    dtype = mlp.gate_proj.weight.dtype
    x = U_rema.to(device=device, dtype=dtype)
    gate = mlp.gate_proj(x.T).T
    up = mlp.up_proj(x.T).T
    h = F.silu(gate) * up
    return torch.nn.functional.normalize(h, dim=0).float()


def get_mom2_covariance(model, tok, layer_name, stats_dir, mom2_dataset="wikipedia"):
    model_name = model.config.name_or_path
    short_model_name = model_name.rstrip("/").split("/")[-1]
    stat = layer_stats(model, tok, layer_name, stats_dir=stats_dir, ds_name=mom2_dataset, to_collect=["mom2"],
                       sample_size=100000, precision="float32", model_name=short_model_name)
    C = stat.mom2.moment().float()
    return C


# --- Main Logic ---

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, default="/data/users/yanrongen/AnyEdit/LLM-Llama-3-8B-Instruct")
    parser.add_argument("--stats_dir", type=str, default="/data/users/yanrongen/ROME-MEMIT/data/stats")
    parser.add_argument("--matrix_dir", type=str, default="rema_matrices")
    # We recommend testing only deep layers, since that is where reasoning takes shape
    parser.add_argument("--layers", nargs='+', type=int, default=[14, 15, 16])
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"Loading model: {args.model_name}")
    try:
        model = AutoModelForCausalLM.from_pretrained(args.model_name, torch_dtype=torch.float16,
                                                     low_cpu_mem_usage=True).to(device).eval()
        tok = AutoTokenizer.from_pretrained(args.model_name)
        tok.pad_token = tok.eos_token
    except Exception as e:
        print(f"❌ Critical Error: {e}")
        return

    # === Core fix: Use the elbow values you determined ===
    # These define the boundary of the "reasoning manifold"
    elbow_configs = {
        "math": 64,  # Elbow point you specified
        "gsm8k": 128  # Elbow point you specified
    }

    sweep_steps = [1, 8, 16, 32, 64, 128, 256, 512]

    print("=" * 70)
    print("⚔️  BIDIRECTIONAL ORTHOGONALITY SCAN (ELBOW-AWARE)")
    print("=" * 70)

    for layer in args.layers:
        print(f"\n📍 Layer {layer}")

        # 1. Prepare MOM2 Full Basis (Sorted by Energy)
        layer_name = f"model.layers.{layer}.mlp.down_proj"
        try:
            C = get_mom2_covariance(model, tok, layer_name, args.stats_dir).to(device)
            vals, vecs = torch.linalg.eigh(C)
            U_mom2_full = torch.flip(vecs, dims=[1])  # [D, D]
        except Exception as e:
            print(f"   ❌ MOM2 Error: {e}")
            continue

        for r_name in ["math", "gsm8k"]:
            elbow_k = elbow_configs[r_name]

            # 2. Prepare REMA Full Basis
            U_rema_raw = safe_load_matrix(f"{args.matrix_dir}/rema_U_{r_name}_L{layer}", device, top_k=512)
            if U_rema_raw is None: continue

            U_rema_full = project_rema_to_mlp(model, layer, U_rema_raw)
            print(f"\n   🛡️ Dataset: {r_name.upper()} (Elbow k={elbow_k})")

            # --- EXPERIMENT A: SAFETY TEST (Fix REMA @ Elbow, Sweep MOM2) ---
            # Logic: Take the full effective reasoning manifold (the first k dimensions).
            # Test: Is this entire reasoning subspace orthogonal to the principal components of the general background?
            print(f"      [Exp A] Fixing REMA (k={elbow_k}), Sweeping MOM2:")
            print(f"      MOM2_K | MaxCos | AvgLeak")

            U_r = U_rema_full[:, :elbow_k]  # use the full Elbow subspace

            for k_m in sweep_steps:
                U_m = U_mom2_full[:, :k_m]
                max_cos, avg_leak = compute_metrics(U_m, U_r)

                # Marker: If it stays orthogonal even with a large background space (k_m=256), that is excellent
                marker = "✅" if max_cos < 0.2 else "⚠️"
                print(f"      {k_m:<6} | {max_cos:.4f} | {avg_leak:.4f} {marker}")

            print("-" * 40)

            # --- EXPERIMENT B: PURITY TEST (Fix MOM2 @ 256, Sweep REMA) ---
            # Logic: Fix a reasonably large general background (MOM2 Top-256).
            # Test: As we introduce more and more reasoning dimensions, when do we start hitting the background wall?
            # Key question: Does it stay orthogonal before reaching the Elbow point (64/128)?
            fixed_mom2_k = 256
            print(f"      [Exp B] Fixing MOM2 (k={fixed_mom2_k}), Sweeping REMA:")
            print(f"      REMA_K | MaxCos | AvgLeak | Status")

            U_m = U_mom2_full[:, :fixed_mom2_k]

            for k_r in sweep_steps:
                if k_r > U_rema_full.shape[1]: break
                U_r = U_rema_full[:, :k_r]
                max_cos, avg_leak = compute_metrics(U_m, U_r)

                status = ""
                if k_r == elbow_k:
                    status = "⬅️ ELBOW"  # Mark the elbow point
                elif k_r < elbow_k and max_cos > 0.3:
                    status = "⚠️ DIRTY EARLY"

                print(f"      {k_r:<6} | {max_cos:.4f} | {avg_leak:.4f} {status}")

            torch.cuda.empty_cache()


if __name__ == "__main__":
    main()