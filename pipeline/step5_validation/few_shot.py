import sys, os
_PROJ_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _PROJ_ROOT not in sys.path:
    sys.path.insert(0, _PROJ_ROOT)

import torch
import argparse
from transformers import AutoModelForCausalLM, AutoTokenizer

# ==============================================================================
# Diagnostic prompt bank (kept in sync with Debug mode)
# ==============================================================================
DEBUG_PROMPTS = {
    "GSM8K (Reasoning)": [
        "Question: Natalia sold clips to 48 of her friends in April, and then she sold half as many clips in May. How many clips did Natalia sell altogether in April and May?\nAnswer:",
        "Question: Janet has 3 times as many marbles as Arnold. If Arnold has 12 marbles, how many marbles do they have together?\nAnswer:",
        "Question: Weng earns $12 an hour for babysitting. Yesterday, she just did 50 minutes of babysitting. How much did she earn?\nAnswer:",
        "Question: James buys a jar of hot sauce for $2. He also buys 5 jars of pickles for $1.5 each. How much does he spend in total?\nAnswer:",
        "Question: A deep-sea monster rises from the bottom of the ocean. It rises 100 feet per minute. If the ocean is 2000 feet deep, how long does it take to reach the surface?\nAnswer:"
    ],
    "MATH (Symbolic)": [
        "Problem: Convert the point $(0,3)$ in rectangular coordinates to polar coordinates.  Enter your answer in the form $(r,\\theta),$ where $r > 0$ and $0 \\le \\theta < 2 \\pi.$",
        "Problem: Define\n\\[p = \\sum_{k = 1}^\\infty \\frac{1}{k^2} \\quad \\text{and} \\quad q = \\sum_{k = 1}^\\infty \\frac{1}{k^3}.\\]Find a way to write\n\\[\\sum_{j = 1}^\\infty \\sum_{k = 1}^\\infty \\frac{1}{(j + k)^3}\\]in terms of $p$ and $q.$",
        "Problem: If $f(x) = \\frac{3x-2}{x-2}$, what is the value of $f(-2) +f(-1)+f(0)$? Express your answer as a common fraction.",
        "Problem: How many positive whole-number divisors does 196 have?",
        "Problem: A regular hexagon can be divided into six equilateral triangles. If the perimeter of one of the triangles is 21 inches, what is the perimeter, in inches, of the regular hexagon?"
    ],
    "MQuAKE (Knowledge)": [
        "Question: Who is the head of state of the country where Ellie Kemper holds a citizenship?",
        "Question: What is the birthplace of the person who created Tetris?",
        "Question: What is the country of citizenship of Marc Cherry?",
        "Question: What is the name of the current head of the Canada government?",
        "Question: Where was LATAM Chile founded?"
    ]
}


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, default="/data/users/yanrongen/AnyEdit/LLM-Llama-3-8B-Instruct")
    return parser.parse_args()


def main():
    args = parse_args()

    print(f"\n{'=' * 60}")
    print(f"🚀 Loading Baseline Model: {args.model_path}")
    print(f"{'=' * 60}")

    # Load model (fp16)
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        device_map="auto",
        torch_dtype=torch.float16
    )
    model.eval()

    print("\n✅ Model Loaded. Running Diagnostic Prompts...\n")

    for category, prompts in DEBUG_PROMPTS.items():
        print(f"\n--- {category} ---")
        for i, prompt_text in enumerate(prompts):
            # Strictly use Llama-3 Chat template, consistent with the experiment code
            full_prompt = (
                f"<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\n"
                f"{prompt_text}"
                f"<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
            )

            inputs = tokenizer(full_prompt, return_tensors="pt").to(model.device)

            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=256,  # Enough tokens to see the beginning of the reasoning
                    do_sample=False,  # Greedy decoding for reproducibility
                    temperature=0.0,
                    pad_token_id=tokenizer.eos_token_id
                )

            output_text = tokenizer.decode(outputs[0], skip_special_tokens=True)

            # Clean the output to show only the generated answer
            if "Answer:" in output_text:
                generated = output_text.split("Answer:")[-1].strip()
            elif "Solution:" in output_text:
                generated = output_text.split("Solution:")[-1].strip()
            elif "assistant" in output_text:
                generated = output_text.split("assistant")[-1].strip()
            else:
                generated = output_text

            # Format for printing
            clean_text = generated.replace('\n', ' ').replace('\r', '')
            print(f"[{i + 1}] {clean_text}")

    print(f"\n{'=' * 60}")
    print("✅ Baseline Check Complete.")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()