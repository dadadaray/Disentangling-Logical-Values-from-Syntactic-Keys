import sys, os
_PROJ_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _PROJ_ROOT not in sys.path:
    sys.path.insert(0, _PROJ_ROOT)

import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os


def plot_custom_heatmap():
    # 1. 加载数据
    try:
        if os.path.exists('rema_overlap_grid.pt'):
            data = torch.load('rema_overlap_grid.pt')
            print("Data loaded successfully from rema_overlap_grid.pt")
        else:
            # 如果文件不存在，生成模拟数据用于演示结构 (Fallback)
            print("Warning: rema_overlap_grid.pt not found. Using mock data for visualization.")
            k_len = 7
            data = {
                6: np.random.uniform(0.3, 0.8, (k_len, k_len)),
                16: np.random.uniform(0.1, 0.5, (k_len, k_len)),  # L16 should have lower overlap
                26: np.random.uniform(0.3, 0.8, (k_len, k_len))
            }
    except Exception as e:
        print(f"Error loading data: {e}")
        return

    layers = [6, 16, 26]
    k_math_list = [1, 2, 4, 8, 16, 32, 64]  # Y轴
    k_gsm_list = [1, 4, 8, 16, 32, 64, 128]  # X轴

    # 设置绘图风格
    sns.set_style("white")
    plt.rcParams['font.family'] = 'DejaVu Sans'

    fig, axes = plt.subplots(1, 3, figsize=(22, 6.5))

    # 使用 YlGnBu (黄-绿-蓝) 渐变
    cmap = "YlGnBu"
    vmin, vmax = 0.0, 1.0

    for idx, layer in enumerate(layers):
        ax = axes[idx]
        matrix = data[layer]

        # 确保矩阵尺寸匹配 (Mock data 或者是真实数据的切片)
        if matrix.shape[0] > 7: matrix = matrix[:7, :]
        if matrix.shape[1] > 7: matrix = matrix[:, :7]

        # 绘制热力图
        heatmap = sns.heatmap(matrix, ax=ax, cmap=cmap, vmin=vmin, vmax=vmax,
                              annot=True, fmt=".2f", annot_kws={"size": 10, "weight": "bold"},
                              cbar=False, square=True, linewidths=1, linecolor='white')

        # 标题
        ax.set_title(f"Layer {layer}", fontsize=18, fontweight='bold', pad=20, color='#333333')

        # X轴标签 (GSM8K)
        ax.set_xticks(np.arange(len(k_gsm_list)) + 0.5)
        ax.set_xticklabels(k_gsm_list, rotation=0, fontsize=12)
        ax.set_xlabel(r"GSM8K Rank ($k_g$)", fontsize=14, fontweight='bold', color='#2ca02c')  # 绿色字体呼应 GSM

        # Y轴标签 (MATH)
        ax.set_yticks(np.arange(len(k_math_list)) + 0.5)
        ax.set_yticklabels(k_math_list, rotation=0, fontsize=12)

        if idx == 0:
            ax.set_ylabel(r"MATH Rank ($k_m$)", fontsize=14, fontweight='bold', color='#1f77b4')  # 蓝色字体呼应 MATH
        else:
            ax.set_ylabel("")
            ax.set_yticks([])

    # 添加 Colorbar
    cbar_ax = fig.add_axes([0.93, 0.15, 0.015, 0.7])
    norm = plt.Normalize(vmin=vmin, vmax=vmax)
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, cax=cbar_ax)
    cbar.set_label("Normalized Subspace Overlap", fontsize=14, rotation=270, labelpad=20)
    cbar.ax.tick_params(labelsize=12)

    plt.suptitle("Geometric Interaction Grid: GSM8K vs. MATH Subspaces", fontsize=22, y=0.98, fontweight='bold')
    plt.subplots_adjust(wspace=0.1, right=0.9)

    plt.savefig("Figure_Overlap_Heatmap_Final.png", bbox_inches='tight', dpi=300)
    plt.show()


if __name__ == "__main__":
    plot_custom_heatmap()