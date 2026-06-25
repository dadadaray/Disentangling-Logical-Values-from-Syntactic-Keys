import sys, os
_PROJ_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _PROJ_ROOT not in sys.path:
    sys.path.insert(0, _PROJ_ROOT)

import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


def plot_paper_figure(data_path, top_k_view=100, cutoff_k=50):
    data = torch.load(data_path)

    eigvals = data["eigenvalues"].numpy()
    proj = data["grammar_projection"].numpy()
    layer = data["layer"]

    # Energy ratios
    eig_ratio = eigvals / eigvals.sum()
    cumulative_energy = np.cumsum(eig_ratio)

    sns.set_theme(style="whitegrid")

    fig, ax1 = plt.subplots(figsize=(10, 6))

    x = np.arange(1, top_k_view + 1)

    # Grammar projection (bars)
    ax1.bar(
        x,
        proj[:top_k_view],
        alpha=0.65,
        color="tab:blue",
        label="BLiMP Grammar Projection",
    )
    ax1.set_xlabel("MOM2 Eigenvector Index (ranked)", fontsize=12)
    ax1.set_ylabel("Grammar Alignment (|cos|)", fontsize=12, color="tab:blue")
    ax1.tick_params(axis="y", labelcolor="tab:blue")

    # Eigenvalue spectrum (line, log)
    ax2 = ax1.twinx()
    ax2.plot(
        x,
        np.log10(eigvals[:top_k_view]),
        color="tab:red",
        linewidth=2.5,
        label="Eigenvalue Spectrum (log)",
    )
    ax2.set_ylabel("Log Eigenvalue Energy", fontsize=12, color="tab:red")
    ax2.tick_params(axis="y", labelcolor="tab:red")

    # Cutoff line
    ax1.axvline(
        cutoff_k,
        linestyle="--",
        linewidth=2,
        color="green",
        label=f"Proposed Top-{cutoff_k} Cutoff",
    )

    # Legend
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper right")

    plt.title(
        f"Grammar Aligns with the Principal Subspace (Layer {layer})",
        fontsize=14,
    )
    plt.tight_layout()
    plt.savefig(f"spectral_analysis_layer{layer}.pdf", dpi=300)
    print(f"Figure saved: spectral_analysis_layer{layer}.pdf")

    # -------------------------
    # Paper-ready statistics
    # -------------------------
    top_k_energy = cumulative_energy[cutoff_k - 1] * 100
    top_k_grammar = proj[:cutoff_k].sum() / proj.sum() * 100

    print("\n--- Statistics for Paper ---")
    print(f"Top-{cutoff_k} eigenvectors capture "
          f"{top_k_energy:.2f}% of total variance.")
    print(f"Top-{cutoff_k} eigenvectors capture "
          f"{top_k_grammar:.2f}% of grammar signal mass.")
    print("This supports that syntax resides in the principal subspace.")


if __name__ == "__main__":
    plot_paper_figure(
        "analysis_results/layer_15_spectral_data.pt",
        top_k_view=100,
        cutoff_k=50,
    )
