import sys, os
_PROJ_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _PROJ_ROOT not in sys.path:
    sys.path.insert(0, _PROJ_ROOT)

import torch
import os
import argparse
import numpy as np


def compute_all_principal_angles(U_a, U_b, max_k=64):

    if U_a is None or U_b is None:
        return np.zeros(max_k)

    if U_a.shape[0] < U_a.shape[1]: U_a = U_a.T
    if U_b.shape[0] < U_b.shape[1]: U_b = U_b.T

    k_effective = min(U_a.shape[1], U_b.shape[1], max_k)
    U_a = U_a[:, :k_effective]
    U_b = U_b[:, :k_effective]

    device = U_a.device
    U_b = U_b.to(device)

    # U_a^T @ U_b
    try:
        interaction = U_a.T @ U_b
        sigmas = torch.linalg.svdvals(interaction)
        return sigmas.cpu().numpy()
    except Exception as e:
        print(f"   ❌ Error during SVD: {e}")
        return np.zeros(max_k)


def safe_load_matrix(path, device):

    if not os.path.exists(path):
        if os.path.exists(path + ".pt"):
            path = path + ".pt"
        else:
            raise FileNotFoundError(f"File not found: {path}")

    try:
        data = torch.load(path, map_location=device)


        if isinstance(data, torch.Tensor):
            return data.float()


        elif isinstance(data, dict):
            possible_keys = ['U', 'V', 'matrix', 'basis', 'components']
            for k in possible_keys:
                if k in data:
                    # print(f"   (Found key '{k}' in dict)")
                    return data[k].float()


            first_key = list(data.keys())[0]
            val = data[first_key]
            if isinstance(val, torch.Tensor):
                print(f"   ⚠️ Warning: Dict found with unknown keys {list(data.keys())}. Using '{first_key}'.")
                return val.float()
            else:
                raise ValueError(f"Dict contains no Tensors. Keys: {data.keys()}")

        else:
            raise ValueError(f"Unknown data type: {type(data)}")

    except Exception as e:
        print(f"   ❌ Error loading {path}: {e}")
        return None


def main():

    matrix_dir = "rema_matrices"
    layers = [12, 13, 14, 15, 16]
    output_file = "radar_chart_data_full.pt"

    device = "cuda" if torch.cuda.is_available() else "cpu"
    results = {}

    print("Collecting Radar Data (Up to Top-64)...")

    for layer in layers:
        print(f"   Scanning Layer {layer}...")

        p_lite = f"{matrix_dir}/rema_U_lite_L{layer}"
        p_gsm = f"{matrix_dir}/rema_U_gsm8k_L{layer}"
        p_math = f"{matrix_dir}/rema_U_math_L{layer}"
        p_noise = f"{matrix_dir}/rema_U_nonsense_L{layer}"

        U_lite = safe_load_matrix(p_lite, device)
        U_gsm = safe_load_matrix(p_gsm, device)
        U_math = safe_load_matrix(p_math, device)
        U_noise = safe_load_matrix(p_noise, device)

        if U_lite is None:
            print("   ⚠️ Skipping layer due to missing REMA-Lite matrix.")
            continue

        # 1. Lite vs GSM
        res_lite_gsm = compute_all_principal_angles(U_lite, U_gsm, max_k=64)

        # 2. Lite vs Math
        res_lite_math = compute_all_principal_angles(U_lite, U_math, max_k=64)

        # 3. GSM vs Math (Baseline 1)
        res_gsm_math = compute_all_principal_angles(U_gsm, U_math, max_k=64)

        # 4. Lite vs Noise (Baseline 2)
        res_lite_noise = compute_all_principal_angles(U_lite, U_noise, max_k=64)

        results[layer] = {
            "Lite-GSM": res_lite_gsm,
            "Lite-Math": res_lite_math,
            "GSM-Math": res_gsm_math,
            "Lite-Noise": res_lite_noise
        }

        if len(res_lite_gsm) > 0:
            print(f"     -> Lite-GSM Top-1 Cosine: {res_lite_gsm[0]:.4f}")

    torch.save(results, output_file)
    print(f"\n✅ Data saved to {output_file}")
    print("Now you can run plot_radar_multi_k.py!")


if __name__ == "__main__":
    main()