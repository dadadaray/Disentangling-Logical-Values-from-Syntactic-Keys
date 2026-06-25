import sys, os
_PROJ_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _PROJ_ROOT not in sys.path:
    sys.path.insert(0, _PROJ_ROOT)

import torch
import os
import argparse
import numpy as np


def load_ortho_matrix(path, k):
    if not os.path.exists(path): return None
    try:
        data = torch.load(path, map_location="cpu")
        U = data.get('U', data.get('projection_matrix', None))
        if U is None: return None
        if U.shape[0] < U.shape[1]: U = U.T
        # Truncate and Normalize
        U = U[:, :k]
        U = torch.nn.functional.normalize(U.float(), p=2, dim=0)
        return U
    except:
        return None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--matrix_dir", type=str, default="rema_matrices")
    args = parser.parse_args()

    layers = [12, 13, 14, 15, 16]

    # User's requested comparison points
    k_math_list = [8, 16, 32, 64]
    k_gsm_list = [32, 64, 128, 256]

    results = {}  # layer -> matrix [len(math), len(gsm)]

    print("Computing Subspace Overlaps...")

    for layer in layers:
        print(f"Processing Layer {layer}...")
        results[layer] = np.zeros((len(k_math_list), len(k_gsm_list)))

        # Load Full Matrices once
        path_math = f"{args.matrix_dir}/rema_U_math_L{layer}.pt"
        path_gsm = f"{args.matrix_dir}/rema_U_gsm8k_L{layer}.pt"

        # We load the MAX k needed to save IO
        U_math_full = load_ortho_matrix(path_math, max(k_math_list))
        U_gsm_full = load_ortho_matrix(path_gsm, max(k_gsm_list))

        if U_math_full is None or U_gsm_full is None:
            print(f"  Missing data for layer {layer}")
            continue

        # Compute Grid
        for i, km in enumerate(k_math_list):
            for j, kg in enumerate(k_gsm_list):
                # Slice
                Um = U_math_full[:, :km]
                Ug = U_gsm_full[:, :kg]

                # Metric: Chordal Distance / Projection Overlap
                # Overlap = || Um^T @ Ug ||_F^2 / min(km, kg)
                # Normalized to [0, 1]
                interaction = torch.matmul(Um.T, Ug)
                overlap = (torch.norm(interaction) ** 2).item()

                # Normalize by the smaller dimension (max possible overlap)
                max_overlap = min(km, kg)
                normalized_score = overlap / max_overlap

                results[layer][i, j] = normalized_score

    torch.save(results, "rema_overlap_grid.pt")
    print("Saved rema_overlap_grid.pt")


if __name__ == "__main__":
    main()