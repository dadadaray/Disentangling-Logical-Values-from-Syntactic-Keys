import argparse
from pathlib import Path
from datasets import load_dataset, load_from_disk, Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from typing import List, Dict


# 假设您的 chunk_long_texts_llama 函数在这里或已从其他地方导入
# ----------------------------------------------------------------------
# 请确保将您之前定义的 chunk_long_texts_llama 函数粘贴到此处
# ----------------------------------------------------------------------
def chunk_long_texts_llama(
        tokenizer: AutoTokenizer,
        raw_dataset: Dataset,
        chunk_size: int = 4096,
        overlap_ratio: float = 0.10
) -> List[Dict]:
    """
    对长文档进行分块，以解决计算协方差矩阵时的 OOM 问题。
    (完整实现请参考之前的回复)
    """
    # 确保 overlap_ratio 在 10% 到 20% 之间
    overlap_ratio = max(0.10, min(0.20, overlap_ratio))
    overlap_tokens = int(chunk_size * overlap_ratio)
    step_size = max(1, chunk_size - overlap_tokens)

    chunked_samples = []

    # 使用 tqdm 追踪进度，因为这很慢
    from tqdm import tqdm
    for example in tqdm(raw_dataset, desc="Chunking and Tokenizing"):
        tokenized_output = tokenizer(
            example['text'],
            add_special_tokens=False,
            return_tensors=None,
            truncation=False
        )
        input_ids = tokenized_output['input_ids']

        # ... [完整的 Chunking 逻辑，使用 step_size 和 chunk_size] ...
        # (简化：直接使用您的原始逻辑)
        start_index = 0
        while start_index < len(input_ids):
            end_index = start_index + chunk_size
            chunk_input_ids = input_ids[start_index:end_index]

            if chunk_input_ids:
                chunked_samples.append({'input_ids': chunk_input_ids})

            start_index += step_size
            if end_index >= len(input_ids):
                break

    return chunked_samples


# ----------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Preprocess and Chunk Wikipedia for MOM2 Calculation")
    parser.add_argument("--model_name",
                        default="/data/jianghc/llama3-8b-instruct",
                        help="Path to the model/tokenizer.")
    parser.add_argument("--output_path",
                        default="data/chunked_wiki_4096",
                        help="Directory to save the chunked dataset.")
    parser.add_argument("--chunk_size",
                        type=int,
                        default=4096,
                        help="The maximum sequence length for each chunk.")
    parser.add_argument("--overlap_ratio",
                        type=float,
                        default=0.10,
                        help="The overlap ratio for chunking.")
    parser.add_argument("--local_wiki_path",
                        default="/data/users/yanrongen/AnyEdit-new/wikipedia",
                        help="Optional path to local Wikipedia cache.")

    args = parser.parse_args()

    # --- 1. 数据集加载逻辑 (与您的 layer_stats 保持一致) ---
    raw_ds_collection = None
    local_path = Path(args.local_wiki_path)

    if local_path.exists():
        print(f"[Info] 尝试加载本地缓存数据集: {args.local_wiki_path}...")
        try:
            raw_ds_train = load_from_disk(str(local_path))
            raw_ds_hf = raw_ds_train
        except Exception as e:
            print(f"[Error] 本地加载失败。错误: {e}")
            raw_ds_hf = load_dataset("wikipedia", "20220301.en", split='train')
    else:
        print(f"[Info] 本地路径未找到，网络下载...")
        raw_ds_hf = load_dataset("wikipedia", "20220301.en", split='train')

    # --- 2. 加载 Tokenizer ---
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)

    # --- 3. 执行分块 (Chunking) ---
    print("\n--- 🚀 开始分词和分块 (CPU 密集型，请耐心等待) ---")
    chunked_data = chunk_long_texts_llama(
        tokenizer,
        raw_ds_hf,
        chunk_size=args.chunk_size,
        overlap_ratio=args.overlap_ratio
    )

    # --- 4. 保存到磁盘 ---
    print(f"\n--- 💾 分块完成，样本总数: {len(chunked_data)} ---")
    ds_chunked = Dataset.from_list(chunked_data)

    output_path = Path(args.output_path)
    output_path.mkdir(parents=True, exist_ok=True)

    ds_chunked.save_to_disk(str(output_path))
    print(f"✅ 预处理数据集已缓存到: {output_path}")


if __name__ == "__main__":
    main()