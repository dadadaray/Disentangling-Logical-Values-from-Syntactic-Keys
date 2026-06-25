import sys, os
_PROJ_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _PROJ_ROOT not in sys.path:
    sys.path.insert(0, _PROJ_ROOT)

import torch
import numpy as np
import os
import argparse


def get_eigenvalues(path):
    if not os.path.exists(path):
        if os.path.exists(path + ".pt"):
            path += ".pt"
        else:
            return None

    try:
        data = torch.load(path, map_location="cpu")

        if isinstance(data, dict):
            if 'eigenvalues' in data:
                return data['eigenvalues'].float().numpy()
            elif 'S' in data:  # SVD singular values

                S = data['S'].float().numpy()
                return S ** 2
            elif 'projection_matrix' in data:

                pass


        if isinstance(data, torch.Tensor):

            if data.shape[0] > data.shape[1]:

                print(f"   (Computing SVD on the fly for {os.path.basename(path)}...)")
                try:
                    S = torch.linalg.svdvals(data.float())
                    return (S ** 2).numpy()
                except:
                    pass
            return None

    except Exception as e:
        print(f"   ❌ Error loading {path}: {e}")
        return None

    return None


def analyze_energy(vals):
    if vals is None or len(vals) == 0:
        return None

    vals = np.sort(vals)[::-1]

    # (Explained Variance Ratio)
    total_energy = np.sum(vals)
    if total_energy == 0: return None

    cum_energy = np.cumsum(vals) / total_energy


    k_80 = np.argmax(cum_energy >= 0.80) + 1
    k_90 = np.argmax(cum_energy >= 0.90) + 1
    k_95 = np.argmax(cum_energy >= 0.95) + 1
    k_99 = np.argmax(cum_energy >= 0.99) + 1

    return {
        "total_dims": len(vals),
        "k_80": int(k_80),
        "k_90": int(k_90),
        "k_95": int(k_95),
        "k_99": int(k_99),
        "curve": cum_energy
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--matrix_dir", type=str, default="rema_matrices")
    parser.add_argument("--output_file", type=str, default="rema_dim_stats.pt")
    args = parser.parse_args()

    layers = [12, 13, 14, 15, 16]
    sources = ["gsm8k", "math", "lite", "nonsense"]

    results = {}

    print(f"Scanning REMA matrices in {args.matrix_dir}...")

    for layer in layers:
        print(f"\n🔍 Analyzing Layer {layer}...")
        results[layer] = {}

        for src in sources:
            filename = f"rema_U_{src}_L{layer}"
            file_path = os.path.join(args.matrix_dir, filename)

            vals = get_eigenvalues(file_path)

            if vals is None:
                print(f"   ⚠️ {src.upper()}: No eigenvalues found.")
                continue

            stats = analyze_energy(vals)
            results[layer][src] = stats

            print(f"   ✅ {src.upper()}: Total={stats['total_dims']}, "
                  f"90%@{stats['k_90']}, 95%@{stats['k_95']}")

    torch.save(results, args.output_file)
    print(f"\n💾 Analysis complete. Data saved to {args.output_file}")
    print("Now upload this file to generate the Scree Plot.")


if __name__ == "__main__":
    main()