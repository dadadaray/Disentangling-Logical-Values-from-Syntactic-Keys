import sys, os
_PROJ_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _PROJ_ROOT not in sys.path:
    sys.path.insert(0, _PROJ_ROOT)

import os
from pathlib import Path
import torch
from datasets import load_dataset, load_from_disk, DatasetDict, Dataset
from tqdm.auto import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import Dataset

from utils.globals import *
from utils.nethook import Trace, set_requires_grad
from utils.runningstats import CombinedStat, Mean, NormMean, SecondMoment, tally
from typing import Generator, Dict

LOCAL_WIKI_PATH = "/data/users/yanrongen/AnyEdit-new/wikipedia"
CHUNK_CACHE_BASE_DIR = Path("/data/users/yanrongen/Anyedit/chunk_cache_datasets")

from utils.tok_dataset import (
    TokenizedDataset,
    dict_to_,
    flatten_masked_batch,
    length_collation,
)

STAT_TYPES = {
    "mom2": SecondMoment,
    "mean": Mean,
    "norm_mean": NormMean,
}


def main():
    """
    Command-line utility to precompute cached stats.
    """
    import argparse

    parser = argparse.ArgumentParser(description="ROME Statistics Collector")

    def aa(*args, **kwargs):
        parser.add_argument(*args, **kwargs)

    aa("--model_name", default="/data/jianghc/llama3-8b-instruct",
       choices=["gpt2-xl", "EleutherAI/gpt-j-6B", "/data/jianghc/llama3-8b-instruct"])
    aa("--dataset", default="wikipedia", choices=["wikitext", "wikipedia"])
    aa("--layers", default=[4, 5, 6, 7, 8], type=lambda x: list(map(int, x.split(","))))
    aa("--to_collect", default=["mom2"], type=lambda x: x.split(","))
    aa("--sample_size", default=100000, type=lambda x: None if x == "all" else int(x))
    aa("--batch_tokens", default=None, type=lambda x: None if x == "any" else int(x))
    aa("--precision", default="float32", choices=["float64", "float32", "float16"])
    aa("--stats_dir", default=STATS_DIR)
    aa("--download", default=1, type=int, choices=[0, 1])
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    model = AutoModelForCausalLM.from_pretrained(args.model_name).eval().cuda()
    set_requires_grad(False, model)

    for layer_num in args.layers:
        print(
            f"Computing stats for layer {layer_num} of {args.model_name} "
            f'over {args.sample_size or "all"} samples of {args.dataset}. '
            "Note, the statistics are collected over the inputs to the second MLP layer, "
            "or equivalently the outputs of the first MLP layer."
        )
        # proj_layer_name = "c_proj" if "gpt2" in args.model_name else "fc_out"
        # layer_name = f"transformer.h.{layer_num}.mlp.{proj_layer_name}"
        layer_name = f"model.layers.{layer_num}.mlp.down_proj"
        layer_stats(
            model,
            tokenizer,
            layer_name,
            args.stats_dir,
            args.dataset,
            args.to_collect,
            sample_size=args.sample_size,
            precision=args.precision,
            batch_tokens=args.batch_tokens,
            download=args.download,
        )


def layer_stats(
        model,
        tokenizer,
        hparams,
        layer_name,
        stats_dir,
        ds_name,
        to_collect,
        all_eval_data=None,  # New optional parameter, only used during RECT
        model_name=None,
        sample_size=None,
        precision=None,
        batch_tokens=None,
        download=True,
        progress=tqdm,
        force_recompute=False
):
    """
    Function to load or compute cached stats.
    """

    # Helper function: safely get the model name
    def _get_safe_model_name(model):
        # Replace special characters in the model path with '_' for use in filenames
        return model.config._name_or_path.replace("/", "_").replace("-", "_").replace(".", "_")

    # ----------------------------------------------------
    # Final corrected get_ds function
    # ----------------------------------------------------
    def get_ds(hparams, tokenizer, model, batch_tokens=None, ds_name=None):
        # ----------------------------------------------------
        # Bug fix: get the dataset name from the ds_name parameter
        # ----------------------------------------------------
        if ds_name is None:
            # If ds_name was not explicitly passed, fall back to trying to get it from hparams (backward compatibility)
            if hasattr(hparams, 'dataset'):
                ds_name = hparams.dataset
            else:
                # If hparams also lacks a 'dataset' attribute, hardcode to 'wikipedia' (or throw an error)
                # Because MOM2 stats default to using wikipedia
                ds_name = "wikipedia"
                print("[Warning] 'dataset' name not found in hparams or arguments. Defaulting to 'wikipedia'.")

        # ... [Function body: the original line ds_name = hparams.dataset is now replaced by the ds_name variable] ...

        # ... [In function body, continue using ds_name variable, e.g.:] ...
        OOM_SAFE_MAXLEN = 4096

        # ----------------------------------------------------
        # 1. Maxlen determination logic (must run first, to determine CHUNK_SIZE)
        # ----------------------------------------------------

        # Original logic: get maxlen from model config
        if hasattr(model.config, 'n_positions'):
            maxlen = model.config.n_positions
        elif hasattr(model.config, 'max_sequence_length'):
            maxlen = model.config.max_sequence_length
        elif hasattr(model.config, 'max_position_embeddings'):
            maxlen = model.config.max_position_embeddings
        elif hasattr(model.config, 'seq_length'):
            maxlen = model.config.seq_length
        else:
            raise NotImplementedError("Unable to determine maxlen from model config.")

        # Final override: if original maxlen exceeds the safe limit, force it to the safe limit
        if maxlen > OOM_SAFE_MAXLEN:
            #is_llama_clamped = True
            maxlen = OOM_SAFE_MAXLEN
            print(f"[Info] 🚨 Force maxlen down to OOM safe limit: {maxlen}")

        # Apply batch_tokens limit
        if batch_tokens is not None and batch_tokens < maxlen:
            maxlen = batch_tokens

        CHUNK_SIZE = maxlen  # Final chunk/sequence length

        # ----------------------------------------------------
        # 2. Dynamically generate model- and chunk-size-dependent cache path
        # ----------------------------------------------------
        model_name_safe = _get_safe_model_name(model)
        ds_name = ds_name
        #chunked_path = CHUNK_CACHE_BASE_DIR / f"{ds_name}_{model_name_safe}_{CHUNK_SIZE}"
        # ----------------------------------------------------
        # 4. Cache miss: load raw dataset
        # ----------------------------------------------------
        valid_mom2_datasets = ["wikipedia", "wikitext"]
        if ds_name not in valid_mom2_datasets:
            print(f"[WARNING] Invalid ds_name '{ds_name}' detected. Forcing to 'wikipedia' for MOM2 stats.")
            ds_name = "wikipedia"

        # ──────────────────────  New caching logic  ──────────────────────
        if ds_name == "wikipedia":
            local_path = Path(LOCAL_WIKI_PATH)  # /…/wikipedia
            cache_path = local_path.parent / "wikipedia_cached_dataset"

            # 1. Cache hit: instant load
            if cache_path.exists():
                print(f"[Cache HIT] Loading merged Wikipedia from {cache_path}")
                raw_ds_hf = load_from_disk(str(cache_path))

            # 2. Cache miss: merge all parquet files and save cache
            else:
                if not local_path.exists():
                    raise FileNotFoundError(f"Local Wikipedia path {local_path} does not exist.")

                print(f"[Cache MISS] Merging {local_path}/*.parquet → {cache_path}")
                # Read all parquet files at once (datasets auto-merges them)
                raw_ds_hf = load_dataset(
                    "parquet",
                    data_files=[str(p) for p in local_path.glob("*.parquet")],
                    split="train"
                )
                # Save as Arrow format for fast load_from_disk
                raw_ds_hf.save_to_disk(str(cache_path))
                print(f"[Cache SAVED] Merged dataset saved to {cache_path}")

        # ──────────────────────  Other datasets unchanged  ──────────────────────
        else:
            config_map = {"wikitext": "wikitext-103-raw-v1"}
            if ds_name == "wikipedia":
                raise ValueError("Wikipedia should be loaded from local path only.")
            config_name = config_map.get(ds_name)
            if config_name is None:
                raise ValueError(f"Invalid ds_name '{ds_name}'. Supported: {list(config_map.keys())}")
            raw_ds_hf = load_dataset(ds_name, config_name, split="train")
        # ----------------------------------------------------
        # 5. Chunking / cache generation logic
        # ----------------------------------------------------
        ds_source = raw_ds_hf

        # ----------------------------------------------------
        # 6. Return TokenizedDataset
        # ----------------------------------------------------
        return TokenizedDataset(ds_source, tokenizer, maxlen=CHUNK_SIZE)

    # Continue with computation of statistics
    #batch_size = 100  # Examine this many dataset texts at once
    batch_size = 5# <--- Force DataLoader batch size to 1 for minimal memory footprint
    if hasattr(model.config, 'n_positions'):
        npos = model.config.n_positions
    elif hasattr(model.config, 'max_sequence_length'):
        npos = model.config.max_sequence_length
    elif hasattr(model.config, 'max_position_embeddings'):
        npos = model.config.max_position_embeddings
    elif hasattr(model.config, 'seq_length'):
        npos = model.config.seq_length
    else:
        raise NotImplementedError

    if hasattr(model.config, 'model_type') and 'mistral' in model.config.model_type:
        if hasattr(model.config, 'sliding_window') and model.config.sliding_window:
            npos = model.config.sliding_window or 4096
        else:
            npos = 4096
    if hasattr(model.config, 'model_type') and 'qwen2' in model.config.model_type:
        npos = 4096
    if batch_tokens is None:
        batch_tokens = npos * 3  # Sort and divide into batches with this many tokens
    if precision is None:
        precision = "float32"
    dtype = getattr(torch, precision)
    size_suffix = "" if sample_size is None else f"_{sample_size}"
    if batch_tokens < npos:
        size_suffix = "_t{batch_tokens}" + size_suffix
    if model_name is None:
        # model_name = model.config._name_or_path.replace("/", "_")
        model_name = model.config._name_or_path.rsplit("/")[-1]

    stats_dir = Path(stats_dir)
    file_extension = f"{model_name}/{ds_name}_stats/{layer_name}_{precision}_{'-'.join(sorted(to_collect))}{size_suffix}.npz"
    filename = stats_dir / file_extension

    print(f"Computing Cov locally....")

    # ds = get_ds() if not filename.exists() else None
    ds = get_ds(hparams, tokenizer, model, ds_name=ds_name) if not filename.exists() else None
    if progress is None:
        progress = lambda x: x

    stat = CombinedStat(**{k: STAT_TYPES[k]() for k in to_collect})
    loader = tally(
        stat,
        ds,
        cache=(filename if not force_recompute else None),
        sample_size=sample_size,
        batch_size=batch_size,
        collate_fn=length_collation(batch_tokens),
        pin_memory=True,
        random_sample=1,
        num_workers=0,
    )
    batch_count = -(-(sample_size or len(ds)) // batch_size)
    with torch.no_grad():
        for batch_group in progress(loader, total=batch_count):
            for batch in batch_group:
                batch = dict_to_(batch, "cuda")
                with Trace(
                        model, layer_name, retain_input=True, retain_output=False, stop=True
                ) as tr:
                    model(**batch)
                feats = flatten_masked_batch(tr.input, batch["attention_mask"])
                # --- OOM fix: force-release GPU memory captured by Trace ---
                del tr.input  # <--- Explicitly delete tr.input
                tr.input = None  # <--- Ensure the reference is nullified
                # feats = flatten_masked_batch(tr.output, batch["attention_mask"])
                feats = feats.to(dtype=dtype).cpu()
                stat.add(feats)
    return stat


if __name__ == "__main__":
    main()
