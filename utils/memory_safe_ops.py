# utils/memory_safe_ops.py
"""
memory_safe_ops.py
------------------
显存友好的运算工具，用于：
  - 分块前向计算 (chunked_forward)
  - CPU 累积协方差 (cpu_accumulate_cov)
  - 安全矩阵乘法 (safe_mm)
  - 动态显存监控
"""

import torch
import gc
from contextlib import contextmanager


def get_model_device(model):
    """返回模型当前所在设备"""
    try:
        return next(model.parameters()).device
    except StopIteration:
        return torch.device("cpu")


@torch.no_grad()
def safe_mm(a: torch.Tensor, b: torch.Tensor, chunk_size: int = 4096):
    """
    分块矩阵乘法，避免单次 OOM。
    a: [N, D]
    b: [D, D]
    返回 [N, D]
    """
    if a.device.type == "cuda" and a.numel() * b.shape[1] * 2e-7 > 0.8:
        print(f"[memory_safe_ops] Using chunked mm: a={a.shape}, b={b.shape}")
    results = []
    for start in range(0, a.size(0), chunk_size):
        end = min(start + chunk_size, a.size(0))
        results.append(a[start:end] @ b)
        torch.cuda.empty_cache()
    return torch.cat(results, dim=0)


def cpu_accumulate_cov(features_iter, dtype=torch.float32):
    """
    在 CPU 上累积协方差矩阵（防止 GPU OOM）
    features_iter: 一个迭代器，每次返回一个 [N, D] 的 torch.Tensor
    """
    mom1 = None
    mom2 = None
    n_seen = 0

    for feats in features_iter:
        feats_cpu = feats.detach().to("cpu", dtype)
        N, D = feats_cpu.shape

        if mom1 is None:
            mom1 = torch.zeros(D, dtype=torch.float64)
            mom2 = torch.zeros((D, D), dtype=torch.float64)

        mom1 += feats_cpu.sum(0).double()
        mom2 += (feats_cpu.T @ feats_cpu).double()
        n_seen += N
        gc.collect()

    if n_seen == 0:
        raise RuntimeError("No features processed in cpu_accumulate_cov")

    mean = (mom1 / n_seen).double()
    cov = (mom2 / n_seen).double() - torch.outer(mean, mean)
    return cov


@torch.no_grad()
def chunked_forward(model, input_ids, attention_mask=None, chunk_size=256):
    """
    将长序列分块前向，以降低显存峰值。
    """
    model_device = get_model_device(model)
    input_ids = input_ids.to(model_device)
    attention_mask = attention_mask.to(model_device) if attention_mask is not None else None

    outputs = []
    for i in range(0, input_ids.size(1), chunk_size):
        out = model(input_ids[:, i:i + chunk_size], attention_mask=attention_mask)
        outputs.append(out.logits.cpu())  # 中间结果移到CPU
        torch.cuda.empty_cache()

    return torch.cat(outputs, dim=1)


@contextmanager
def temporary_offload(model, device="cpu"):
    """
    临时将模型权重搬到 CPU，再搬回。
    用于低显存时做一次性计算。
    """
    original_device = get_model_device(model)
    model.to(device)
    try:
        yield model
    finally:
        model.to(original_device)
