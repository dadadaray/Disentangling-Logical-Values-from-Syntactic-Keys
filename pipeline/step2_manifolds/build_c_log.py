import torch
import os
import numpy as np
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
from utils import nethook
import sys

sys.path.append(".")

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from data.gsm8k import GSM8KDataset
from data.math import MATHDataset

MODEL_PATH = "/data/users/yanrongen/AnyEdit/LLM-Qwen2.5-7B"
LAYERS = [6,16,26]
DATASET_NAME = "gsm8k"
DATA_DIR = os.path.join(project_root, "data")
N_SAMPLES = 1319
SAVE_DIR = os.path.join(project_root, "rema_matrices_qwen")


def get_llama_template(q):
    return f"<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\n{q}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"


def main():
    print(f"Loading model: {MODEL_PATH}")
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH,
        device_map="auto",
        torch_dtype=torch.float32,
        low_cpu_mem_usage=True
    ).eval()
    tok = AutoTokenizer.from_pretrained(MODEL_PATH, padding_side='left')
    tok.pad_token = tok.eos_token

    first_param_device = next(model.parameters()).device
    print(f"✅ Model loaded. Primary device: {first_param_device}")

    if first_param_device.type == 'cpu' and torch.cuda.is_available():
        print("❌ CRITICAL ERROR 100: Model failed to load on GPU (likely VRAM busy).")
        print("   -> Exiting immediately to trigger Bash retry logic.")
        sys.exit(100)


    print(f"Loading dataset: {DATASET_NAME}...")
    if DATASET_NAME == "gsm8k":
        dataset = GSM8KDataset(DATA_DIR, MODEL_PATH, size=None)
    elif DATASET_NAME == "math":
        dataset = MATHDataset(DATA_DIR, MODEL_PATH, size=None)
    else:
        raise ValueError("Unknown dataset")

    print(f"Starting REMA Collection ({N_SAMPLES} samples)...")
    collected_states = {l: [] for l in LAYERS}
    count = 0

    pbar = tqdm(total=N_SAMPLES, desc="Collecting...")

    for i in range(len(dataset)):
        if count >= N_SAMPLES: break

        item = dataset[i]
        question = item['question']
        ground_truth = item['answer']

        if "prompt" in item and item["prompt"]:
            inp_text = item["prompt"]
        else:
            inp_text = get_llama_template(f"{question}\nLet's think step by step.")

        inputs = tok(inp_text, return_tensors="pt").to(first_param_device)

        try:
            with torch.no_grad():
                output_ids = model.generate(**inputs, max_new_tokens=512, do_sample=False)
        except Exception as e:
            print(f"Gen Error: {e}")
            continue

        pred_text = tok.decode(output_ids[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)

        is_correct = False
        if DATASET_NAME == "math":
            if ground_truth.replace(" ", "") in pred_text.replace(" ", ""): is_correct = True
        else:
            if ground_truth in pred_text: is_correct = True

        if is_correct:
            count += 1
            pbar.update(1)

            gen_len = output_ids.shape[1] - inputs.input_ids.shape[1]

            if gen_len > 0:
                with torch.no_grad():
                    with nethook.TraceDict(model, [f"model.layers.{l}" for l in LAYERS]) as tr:
                        model(output_ids)

                for l in LAYERS:
                    # tr.output: [1, Total_Seq, Dim]
                    layer_out = tr[f"model.layers.{l}"].output
                    if isinstance(layer_out, tuple):
                        layer_out = layer_out[0]

                    # 3D [Batch, Seq, Dim]
                    if layer_out.ndim == 3:
                        layer_out = layer_out[0]

                    actual_len = layer_out.shape[0]
                    safe_gen_len = min(gen_len, actual_len)

                    # Reasoning Trajectory
                    reasoning_states = layer_out[-safe_gen_len:, :]  # [Gen_Seq, Dim]

                    # Mean Pooling
                    vec = reasoning_states.mean(dim=0).cpu()
                    collected_states[l].append(vec)

    pbar.close()
    print(f"Collection complete. Found {count} correct samples.")

    if not os.path.exists(SAVE_DIR): os.makedirs(SAVE_DIR)

    for l in LAYERS:
        if not collected_states[l]: continue

        #[N, Dim]
        X = torch.stack(collected_states[l]).float()

        # PCA
        mean = X.mean(dim=0)
        X_centered = X - mean
        k = min(4096, X.shape[0])
        _, S, V = torch.svd_lowrank(X_centered, q=k)

        save_data = {
            "projection_matrix": V,  # [Dim, k]
            "mean_vector": mean,  # [Dim]
            "points_sample": X,  # [N, Dim]
            "eigenvalues": S  # [k]
        }

        save_path = f"{SAVE_DIR}/rema_U_{DATASET_NAME}_L{l}.pt"
        torch.save(save_data, save_path)
        print(f"Saved full manifold data for L{l} -> {save_path}")


if __name__ == "__main__":
    main()