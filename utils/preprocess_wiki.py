import argparse
from pathlib import Path
from datasets import load_dataset, load_from_disk, Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from typing import List, Dict


# Assumes chunk_long_texts_llama is defined here or imported from elsewhere
# ----------------------------------------------------------------------
# Make sure to paste your chunk_long_texts_llama function definition here
# ----------------------------------------------------------------------
def chunk_long_texts_llama(
        tokenizer: AutoTokenizer,
        raw_dataset: Dataset,
        chunk_size: int = 4096,
        overlap_ratio: float = 0.10
) -> List[Dict]:
    """
    Chunks long documents to solve OOM issues when computing the covariance matrix.
    (See previous reply for the full implementation)
    """
    # Clamp overlap_ratio between 10% and 20%
    overlap_ratio = max(0.10, min(0.20, overlap_ratio))
    overlap_tokens = int(chunk_size * overlap_ratio)
    step_size = max(1, chunk_size - overlap_tokens)

    chunked_samples = []

    # Use tqdm to track progress, since this is slow
    from tqdm import tqdm
    for example in tqdm(raw_dataset, desc="Chunking and Tokenizing"):
        tokenized_output = tokenizer(
            example['text'],
            add_special_tokens=False,
            return_tensors=None,
            truncation=False
        )
        input_ids = tokenized_output['input_ids']

        # ... [Full chunking logic using step_size and chunk_size] ...
        # (Simplified: using the original logic directly)
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

    # --- 1. Dataset loading logic (consistent with your layer_stats) ---
    raw_ds_collection = None
    local_path = Path(args.local_wiki_path)

    if local_path.exists():
        print(f"[Info] Attempting to load local cached dataset: {args.local_wiki_path}...")
        try:
            raw_ds_train = load_from_disk(str(local_path))
            raw_ds_hf = raw_ds_train
        except Exception as e:
            print(f"[Error] Local load failed. Error: {e}")
            raw_ds_hf = load_dataset("wikipedia", "20220301.en", split='train')
    else:
        print(f"[Info] Local path not found, downloading from web...")
        raw_ds_hf = load_dataset("wikipedia", "20220301.en", split='train')

    # --- 2. Load Tokenizer ---
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)

    # --- 3. Run Chunking ---
    print("\n--- Starting tokenization and chunking (CPU-bound, please be patient) ---")
    chunked_data = chunk_long_texts_llama(
        tokenizer,
        raw_ds_hf,
        chunk_size=args.chunk_size,
        overlap_ratio=args.overlap_ratio
    )

    # --- 4. Save to disk ---
    print(f"\n--- Chunking complete, total samples: {len(chunked_data)} ---")
    ds_chunked = Dataset.from_list(chunked_data)

    output_path = Path(args.output_path)
    output_path.mkdir(parents=True, exist_ok=True)

    ds_chunked.save_to_disk(str(output_path))
    print(f"Preprocessed dataset cached to: {output_path}")


if __name__ == "__main__":
    main()