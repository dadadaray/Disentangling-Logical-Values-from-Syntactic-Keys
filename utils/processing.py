import json
import os
import re
from tqdm import tqdm

# ================= Configuration =================
# Make sure your original jsonl file is at this path
MATH_INPUT_PATH = "D:\py demos\LLM\AnyEdit-main\AnyEdit-main/data/math/test.jsonl"
MATH_OUTPUT_PATH = "data/math/test.json"


# =================================================

def extract_boxed_answer(solution):
    """
    Robustly extracts the content of LaTeX \boxed{...}, supporting nested braces.
    Example: "The answer is \boxed{\frac{1}{2}}" -> "\frac{1}{2}"
    """
    # 1. Find all occurrences of \boxed{
    # We typically take the last \boxed, since earlier ones may be intermediate steps
    start_indices = [m.start() for m in re.finditer(r'\\boxed\{', solution)]

    if not start_indices:
        return None

    # Take the last \boxed
    start_index = start_indices[-1]

    # 2. Stack-based scan to match the corresponding closing brace }
    content_start = start_index + 7  # len("\\boxed{") == 7
    balance = 0
    answer = []

    for i in range(content_start, len(solution)):
        char = solution[i]

        if char == '{':
            balance += 1
        elif char == '}':
            if balance == 0:
                # Found the outermost closing brace
                return "".join(answer)
            balance -= 1

        answer.append(char)

    return "".join(answer)  # Fallback: return whatever was extracted if unmatched


def process_math_file():
    print(f"Reprocessing MATH dataset from {MATH_INPUT_PATH}...")

    if not os.path.exists(MATH_INPUT_PATH):
        print(f"File not found: {MATH_INPUT_PATH}")
        return

    processed_data = []

    with open(MATH_INPUT_PATH, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    for i, line in enumerate(tqdm(lines, desc="Processing")):
        if not line.strip(): continue
        try:
            item = json.loads(line)

            # MATH raw fields are typically problem, solution
            question = item.get("problem", item.get("question", ""))
            solution = item.get("solution", item.get("full_solution", ""))

            # 1. Prefer the dataset's own answer field (if present and well-formed)
            # If the original file has an "answer" field, using it directly is the most accurate!
            raw_answer = item.get("answer")

            # 2. If absent or suspect, extract the boxed answer from the solution
            extracted_answer = extract_boxed_answer(solution)

            # Decision: if raw_answer looks short (pure answer), use it; otherwise use the boxed one
            # Here we trust the boxed extraction result, as it more precisely matches REMA's approach
            final_answer = extracted_answer if extracted_answer else raw_answer

            if not final_answer:
                final_answer = "No Answer Found"  # Fallback

            new_item = {
                "case_id": i,
                "id": i,
                "question": question,
                "answer": final_answer,  # The answer here is now complete LaTeX
                "full_solution": solution,
                "subject": item.get("subject", "math"),
                "level": item.get("level", "unknown")
            }
            processed_data.append(new_item)

        except json.JSONDecodeError:
            continue

    # Save
    with open(MATH_OUTPUT_PATH, 'w', encoding='utf-8') as f:
        json.dump(processed_data, f, indent=2, ensure_ascii=False)

    print(f"Saved {len(processed_data)} fixed items to {MATH_OUTPUT_PATH}")

    # Verify a few samples
    print("\n--- Sample Check ---")
    for j in range(min(3, len(processed_data))):
        print(f"Q: {processed_data[j]['question'][:50]}...")
        print(f"A: {processed_data[j]['answer']}")
        print("-" * 20)


if __name__ == "__main__":
    process_math_file()