import sys, os
_PROJ_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _PROJ_ROOT not in sys.path:
    sys.path.insert(0, _PROJ_ROOT)

import torch
import os
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer
from utils import nethook

# ================= Configuration =================
MODEL_NAME = "/data/users/yanrongen/AnyEdit/LLM-Llama-3-8B-Instruct"
LAYERS = [12, 13, 14, 15, 16]
SAVE_DIR = "rema_matrices"


# ===========================================

def get_llama_template(q):
    return f"<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\n{q}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"


def main():
    print(f"Loading model: {MODEL_NAME}")
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        device_map="auto",
        torch_dtype=torch.float16,
        low_cpu_mem_usage=True
    ).eval()
    tok = AutoTokenizer.from_pretrained(MODEL_NAME, padding_side='left')
    tok.pad_token = tok.eos_token

    """
        REMA-Lite: Enhanced Version
        Build a general reasoning manifold based on academically validated CoT Triggers and Instruction Templates.
        References: Kojima et al. (2022), Wang et al. (2023), Zhou et al. (2023).
        """

    # 1. Basic Chain-of-Thought
    base_cot = [
        "Let's think step by step.",
        "Let's work this out in a step by step way to be sure we have the right answer.",
        "Let's think about this logically.",
        "Let's solve this problem by splitting it into steps.",
        "First, we need to understand the question. Then, we can solve it.",
        "Let's analyze the given information.",
        "Let's break down the problem.",
        "We need to proceed methodically.",
        "Let's do this one step at a time.",
        "Take a deep breath and work on this problem step-by-step."
    ]

    # 2. Planning & Execution (Plan-and-Solve)
    planning = [
        "Let's make a plan to solve this.",
        "First, create a step-by-step plan. Then, carry out the plan.",
        "Identify the core issue and propose a solution.",
        "Let's list the knowns and unknowns.",
        "What is the first step? What comes next?",
        "Let's organize our thoughts before answering.",
        "Construct a logical argument.",
        "Let's derive the solution from the premises.",
        "We should verify each step of our reasoning.",
        "Let's calculate the intermediate values."
    ]

    # 3. Causal & Counterfactual
    causal = [
        "If X happens, what is the likely consequence?",
        "Because of A, B occurred. Therefore,",
        "This implies a cause-and-effect relationship.",
        "Let's trace the chain of events.",
        "Consider the opposite scenario. What would happen?",
        "The underlying reason for this is",
        "Let's analyze the consequences of this action.",
        "This conclusion follows from the fact that",
        "Let's rule out impossible scenarios.",
        "Hypothetically, if the condition were changed, then"
    ]

    # 4. Math & Symbolic
    math_logic = [
        "To solve for x, we need to",
        "The formula for this calculation is",
        "Let's apply the theorem to this case.",
        "1, 1, 2, 3, 5... The pattern indicates",
        "Given the equation, we can deduce",
        "Let's verify the arithmetic.",
        "The probability of this event is",
        "Let's integrate this information.",
        "By substitution, we get",
        "The geometric properties suggest that"
    ]

    # 5. Critical Thinking
    critical = [
        "Are there any logical fallacies in this argument?",
        "Let's evaluate the strength of this evidence.",
        "Is there a counter-argument?",
        "Let's critique this reasoning.",
        "What is the hidden assumption here?",
        "Let's compare the pros and cons.",
        "Does the conclusion logically follow from the premises?",
        "Let's check for consistency.",
        "Is this explanation plausible?",
        "Let's justify this step."
    ]

    raw_templates = base_cot + planning + causal + math_logic + critical

    prompts = [get_llama_template(t) for t in raw_templates * 5]

    print(f"🚀 Collecting Lite Manifold from {len(prompts)} generic prompts...")

    collected_vecs = {l: [] for l in LAYERS}

    batch_size = 8
    for i in range(0, len(prompts), batch_size):
        batch = prompts[i: i + batch_size]
        device = next(model.parameters()).device
        inputs = tok(batch, return_tensors="pt", padding=True, truncation=True).to(device)

        with torch.no_grad():
            with nethook.TraceDict(model, [f"model.layers.{l}" for l in LAYERS]) as tr:
                model(**inputs)

            for l in LAYERS:
                # [Batch, Seq, Dim]
                out = tr[f"model.layers.{l}"].output
                if isinstance(out, tuple):
                    layer_out = out[0]
                else:
                    layer_out = out

                if layer_out.ndim == 2:
                    layer_out = layer_out.unsqueeze(0)

                target_device = layer_out.device
                mask = inputs["attention_mask"].to(target_device).bool()

                if layer_out.shape[0] != mask.shape[0]:
                    print(f"Error: Shape mismatch at layer {l}. Out: {layer_out.shape}, Mask: {mask.shape}")
                    continue

                valid_acts = layer_out[mask]  # Flatten -> [Total_Valid_Tokens, Dim]
                collected_vecs[l].append(valid_acts.cpu())

    if not os.path.exists(SAVE_DIR): os.makedirs(SAVE_DIR)

    for l in LAYERS:
        if not collected_vecs[l]: continue

        # [Total_Tokens, Dim]
        X = torch.cat(collected_vecs[l], dim=0).float()

        # PCA: Center Data
        mean = X.mean(dim=0)
        X_centered = X - mean

        # SVD
        k = min(4096, X.shape[0])
        try:
            _, _, V = torch.svd_lowrank(X_centered, q=k)
        except Exception as e:
            print(f"SVD failed for L{l}: {e}")
            continue

        save_path = f"{SAVE_DIR}/rema_U_lite_L{l}.pt"
        torch.save(V, save_path)
        print(f"💾 Saved Lite Matrix L{l} -> {save_path}")


if __name__ == "__main__":
    main()