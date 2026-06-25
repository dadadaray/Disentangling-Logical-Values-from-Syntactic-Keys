# utils/model_dispatcher.py
"""
model_dispatcher.py
-------------------
安全加载大模型（Llama/Qwen/...）的通用调度模块。
改进：
  - 支持 Slurm 分配 GPU（非连续 ID）
  - 自动计算每张卡可用显存，避免 OOM
  - 支持 Accelerate offload（CPU/Disk）
  - dispatch=False 使用 transformers 原生 device_map='auto'
  - 关键层（LM head）不放到 meta/CPU
"""

import os
from pathlib import Path
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from accelerate import init_empty_weights, load_checkpoint_and_dispatch

def load_model_and_tokenizer(
    model_name_or_path: str,
    torch_dtype=torch.float16,
    dispatch: bool = True,
    offload_folder: str = "./offload",
    low_cpu_mem_usage: bool = True,
    trust_remote_code: bool = True,
    max_memory: dict = None,
):
    """
    加载大模型并安全分配设备。
    dispatch=True → 使用 accelerate 动态分配
    dispatch=False → 普通加载（device_map='auto')
    """

    print(f"[model_dispatcher] Loading model: {model_name_or_path}")

    # ----------------- Tokenizer -----------------
    tokenizer = AutoTokenizer.from_pretrained(
        model_name_or_path,
        trust_remote_code=trust_remote_code,
        padding_side="left"
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # ----------------- GPU/显存配置 -----------------
    if dispatch:
        # Slurm GPU 优先
        slurm_gpus = None
        if "CUDA_VISIBLE_DEVICES" in os.environ:
            slurm_gpus = [int(x) for x in os.environ["CUDA_VISIBLE_DEVICES"].split(",")]
        elif "SLURM_JOB_GPUS" in os.environ:
            slurm_gpus = [int(x) for x in os.environ["SLURM_JOB_GPUS"].split(",")]

        if torch.cuda.is_available() and slurm_gpus:
            max_memory = {}
            for dev in slurm_gpus:
                props = torch.cuda.get_device_properties(dev)
                max_mem_gib = int(props.total_memory * 0.8 / 1024**3)
                max_memory[dev] = f"{max_mem_gib}GiB"
            max_memory["cpu"] = "64GiB"
        elif torch.cuda.is_available():
            # fallback: 全部 GPU
            max_memory = {}
            for i in range(torch.cuda.device_count()):
                props = torch.cuda.get_device_properties(i)
                max_mem_gib = int(props.total_memory * 0.8 / 1024**3)
                max_memory[i] = f"{max_mem_gib}GiB"
            max_memory["cpu"] = "64GiB"
        else:
            max_memory = {"cpu": "128GiB"}

        # Offload 文件夹
        offload_folder = Path(offload_folder)
        offload_folder.mkdir(exist_ok=True, parents=True)

        # ----------------- Accelerate 加载 -----------------
        print("[model_dispatcher] Using accelerate dispatch mode.")
        with init_empty_weights():
            model = AutoModelForCausalLM.from_pretrained(
                model_name_or_path,
                torch_dtype=torch_dtype,
                low_cpu_mem_usage=low_cpu_mem_usage,
                trust_remote_code=trust_remote_code,
            )
        model.tie_weights()

        # 保证 LM head 不会放到 meta
        no_split_modules = ["LlamaDecoderLayer", "QwenBlock"]
        if hasattr(model, "lm_head"):
            no_split_modules.append("lm_head")

        model = load_checkpoint_and_dispatch(
            model,
            model_name_or_path,
            device_map="auto",  # 根据 max_memory 自动分片
            max_memory=max_memory,
            no_split_module_classes=no_split_modules,
            offload_folder=str(offload_folder),
            dtype=torch_dtype,
        )

    else:
        # ----------------- transformers 原生 device_map -----------------
        print("[model_dispatcher] Using transformers auto device_map.")
        model = AutoModelForCausalLM.from_pretrained(
            model_name_or_path,
            torch_dtype=torch_dtype,
            device_map="auto",
            low_cpu_mem_usage=low_cpu_mem_usage,
            trust_remote_code=trust_remote_code,
        )

    model.eval()

    # 记录模型当前所在设备
    model_device = next(model.parameters()).device
    print(f"[model_dispatcher] Model initialized on device: {model_device}")
    print(f"[model_dispatcher] Max memory per device: {max_memory}")

    return model, tokenizer
