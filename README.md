# IntIso: Interaction Isolation for Reasoning-Safe Knowledge Editing

Official code for the TACL paper: **"Disentangling Logical Values from Syntactic Keys: Exposing Geometric Conflict in Knowledge Editing"**

## Overview

Knowledge editing (KE) efficiently updates large language models without retraining, but standard covariance-based methods fail to sustain multi-hop reasoning chains. We trace this *structural collapse* to a fundamental **geometric conflict** within reasoning hubs: dense syntactic features structurally suppress sparse logical reasoning pathways.

**Interaction Isolation (IntIso)** is a geometric analysis framework that disentangles logical values from syntactic keys via spectral truncation, enabling reasoning-safe knowledge editing through task-specific operations (ablation for verbose reasoning, injection for symbolic reasoning).

## Repository Structure

```
IntIso/
├── paper/                     # LaTeX source of the paper
├── pipeline/                  # Experiment scripts, ordered by paper section
│   ├── step1_localization/    # Sec 3.1: Causal tracing of reasoning hubs
│   ├── step2_manifolds/       # Sec 3.2: C_syn and C_log manifold construction
│   ├── step3_verification/    # Sec 3.3: Geometric conflict verification
│   ├── step4_intiso/          # Sec 4: IntIso ablation/injection
│   └── step5_validation/      # Sec 5: Figures and diagnostics
├── slurm/                     # SLURM job scripts with exact experimental parameters
│   └── step*/                 # Mirrors pipeline/ structure
├── data/                      # Dataset loaders (GSM8K, MATH)
├── utils/                     # Core utilities + data preprocessing tools
└── notebooks/                 # Causal tracing & visualization notebooks
```

## Pipeline

| Step | Paper Section | Key Scripts | Purpose |
|------|-------------|-------------|---------|
| 1 | Sec 3.1 | `gradient_probe.py`, `measure_sensitivity.py` | Localize reasoning hubs via causal tracing |
| 2 | Sec 3.2 | `build_c_syn.py`, `build_c_log.py`, `build_c_log_lite.py`, `build_null_baseline.py`, `analyze_eigenvalues.py`, `analyze_curvature_*.py` | Construct syntactic (C_syn) and logical (C_log) manifolds |
| 3 | Sec 3.3 | `orthogonality_test.py`, `spectrum_blimp_mom2.py`, `spectrum_blimp_rema.py`, `subspace_compare.py`, `subspace_overlap.py` | Verify geometric conflict between manifolds |
| 4 | Sec 4 | `ablation.py` | IntIso: cross-manifold operator with ablation/injection |
| 5 | Sec 5 | `plot_analysis.py`, `draw.py`, `few_shot.py` | Generate figures and diagnostic outputs |

See [PAPER_PIPELINE.md](PAPER_PIPELINE.md) for detailed paper-section-to-code mapping.

## Requirements

- **GPU**: A100 80GB or RTX 4090 (experiments were run on SLURM cluster)
- Python 3.10+, PyTorch 1.12+

```bash
pip install -r requirements.txt
```

## Quick Start

### Setup PYTHONPATH
```bash
export PYTHONPATH=$(pwd):$PYTHONPATH
```

### Run IntIso Experiment (Sec 4)
```bash
python pipeline/step4_intiso/ablation.py \
    --model_path /path/to/Llama-3-8B-Instruct \
    --rema_dir /path/to/rema_matrices \
    --mom2_dir /path/to/mom2_eig \
    --data_root /path/to/data \
    --layer 16 \
    --rema_type math --rema_k 64 --mom2_k 128 \
    --tasks gsm8k math blimp \
    --run_gsm8k_sweep
```

### Build Manifolds (Sec 3.2)
```bash
# C_syn from Wikipedia
python pipeline/step2_manifolds/build_c_syn.py --model_name meta-llama/Meta-Llama-3-8B-Instruct

# C_log from reasoning traces
python pipeline/step2_manifolds/build_c_log.py --model_name meta-llama/Meta-Llama-3-8B-Instruct --dataset gsm8k
```

### Run Verification (Sec 3.3)
```bash
python pipeline/step3_verification/orthogonality_test.py \
    --model_name meta-llama/Meta-Llama-3-8B-Instruct \
    --stats_dir /path/to/stats --matrix_dir /path/to/rema_matrices
```

SLURM submission scripts (`.sh` files) with exact parameter configurations used in the paper are in the `slurm/` directory.

## Models

Experiments were conducted on:
- **Llama-3-8B-Instruct** (primary case study)
- **Qwen-2.5-7B-Instruct** (cross-model validation)

## Datasets

Required datasets (not included, download separately):
- **Wikipedia** (for C_syn / MOM2 statistics)
- **GSM8K** (verbatim reasoning)
- **MATH-500** (symbolic reasoning)
- **MQuAKE-CF** (multi-hop reasoning probe)
- **BLiMP** (grammatical minimal pairs)

## Citation

```bibtex
@article{intiso2026,
  title={Disentangling Logical Values from Syntactic Keys: Exposing Geometric Conflict in Knowledge Editing},
  author={...},
  journal={Transactions of the Association for Computational Linguistics},
  year={2026}
}
```

## License

See [LICENSE](LICENSE).
