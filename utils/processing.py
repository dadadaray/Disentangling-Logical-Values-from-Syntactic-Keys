import json
import os
import re
from tqdm import tqdm

# ================= 配置 =================
# 请确保你的原始 jsonl 文件在这个路径
MATH_INPUT_PATH = "D:\py demos\LLM\AnyEdit-main\AnyEdit-main/data/math/test.jsonl"
MATH_OUTPUT_PATH = "data/math/test.json"


# =======================================

def extract_boxed_answer(solution):
    """
    鲁棒地提取 LaTeX 中 \boxed{...} 的内容，支持嵌套括号。
    例如: "The answer is \boxed{\frac{1}{2}}" -> "\frac{1}{2}"
    """
    # 1. 找到所有 \boxed{ 的起始位置
    # 我们通常取最后一个 \boxed，因为前面的可能是中间步骤
    start_indices = [m.start() for m in re.finditer(r'\\boxed\{', solution)]

    if not start_indices:
        return None

    # 取最后一个 \boxed
    start_index = start_indices[-1]

    # 2. 栈式扫描，匹配对应的结束括号 }
    content_start = start_index + 7  # len("\\boxed{") == 7
    balance = 0
    answer = []

    for i in range(content_start, len(solution)):
        char = solution[i]

        if char == '{':
            balance += 1
        elif char == '}':
            if balance == 0:
                # 找到了最外层的结束括号
                return "".join(answer)
            balance -= 1

        answer.append(char)

    return "".join(answer)  # 如果没闭合，返回当前提取到的所有内容(兜底)


def process_math_file():
    print(f"🔄 Reprocessing MATH dataset from {MATH_INPUT_PATH}...")

    if not os.path.exists(MATH_INPUT_PATH):
        print(f"❌ File not found: {MATH_INPUT_PATH}")
        return

    processed_data = []

    with open(MATH_INPUT_PATH, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    for i, line in enumerate(tqdm(lines, desc="Processing")):
        if not line.strip(): continue
        try:
            item = json.loads(line)

            # MATH 原始字段通常是 problem, solution
            question = item.get("problem", item.get("question", ""))
            solution = item.get("solution", item.get("full_solution", ""))

            # 1. 优先使用数据集自带的 answer (如果有且格式正确)
            # 你贴的样本里有 "answer" 字段，如果原始文件里有，直接用它最准！
            raw_answer = item.get("answer")

            # 2. 如果没有或看起来不对，从 solution 提取 boxed
            extracted_answer = extract_boxed_answer(solution)

            # 决策：如果 raw_answer 看起来很短（纯答案），用它；否则用 boxed
            # 这里我们信任 boxed 提取的结果，因为它更精准对应 REMA 的做法
            final_answer = extracted_answer if extracted_answer else raw_answer

            if not final_answer:
                final_answer = "No Answer Found"  # 兜底

            new_item = {
                "case_id": i,
                "id": i,
                "question": question,
                "answer": final_answer,  # 这里的答案现在是完整的 LaTeX
                "full_solution": solution,
                "subject": item.get("subject", "math"),
                "level": item.get("level", "unknown")
            }
            processed_data.append(new_item)

        except json.JSONDecodeError:
            continue

    # 保存
    with open(MATH_OUTPUT_PATH, 'w', encoding='utf-8') as f:
        json.dump(processed_data, f, indent=2, ensure_ascii=False)

    print(f"✅ Saved {len(processed_data)} fixed items to {MATH_OUTPUT_PATH}")

    # 验证几个样本
    print("\n--- Sample Check ---")
    for j in range(min(3, len(processed_data))):
        print(f"Q: {processed_data[j]['question'][:50]}...")
        print(f"A: {processed_data[j]['answer']}")
        print("-" * 20)


if __name__ == "__main__":
    process_math_file()