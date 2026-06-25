# TACL Paper: Code-to-Section Mapping

**Paper**: "Disentangling Logical Values from Syntactic Keys: Exposing Geometric Conflict in Knowledge Editing"

---

## Section 3.1 — Localization of the Reasoning Hub

**Goal**: Causal tracing to identify layers where multi-hop reasoning diverges from factual storage.

| Script | Purpose | Key Output |
|--------|---------|------------|
| `pipeline/step1_localization/gradient_probe.py` | Causal tracing via gradient-based probing. Corrupts subject embeddings with Gaussian noise, patches per-layer hidden states to clean values, measures answer-logit restoration. Differentiates single-hop vs. multi-hop recovery. | `results/causal_probe/probe_results_{model}.pt` |
| `pipeline/step1_localization/measure_sensitivity.py` | REMA signal-vs-noise validation. Applies REMA perturbation and energy-matched random perturbation to `down_proj` weight. Measures MQuAKE masked loss change per layer. | Console table: Signal Sensitivity, Noise Sensitivity, SNR Ratio per layer |

**Key finding**: Llama-3 centralizes reasoning in Layers 10-16; Qwen-2.5 shifts to >25.

---

## Section 3.2 — Manifold Construction

**Goal**: Build the syntactic manifold (C_syn) from Wikipedia and logical manifold (C_log) from reasoning traces.

### C_syn (Syntactic Manifold / MOM2)
| Script | Purpose | Data Source |
|--------|---------|-------------|
| `pipeline/step2_manifolds/build_c_syn.py` | Computes uncentered second moment (MOM2) over Wikipedia text at `down_proj` input. Performs the eigendecomposition that defines the Syntactic Key Basis (K_syn). | Wikipedia |

### C_log (Logical Manifold / REMA)
| Script | Purpose | Data Source |
|--------|---------|-------------|
| `pipeline/step2_manifolds/build_c_log.py` | Runs model on GSM8K/MATH, collects hidden states from correctly-solved instances, applies SVD (PCA) to extract C_log basis vectors. | GSM8K / MATH |
| `pipeline/step2_manifolds/build_c_log_lite.py` | Constructs a "lite" generic reasoning manifold from hand-crafted CoT/planning/math templates (not task-specific data). | 50 human-written reasoning templates |
| `pipeline/step2_manifolds/build_null_baseline.py` | Generates a "nonsense" manifold from random-word prompts as a null baseline for noise quantification. | Random word sequences |

### Eigenvalue Analysis
| Script | Purpose |
|--------|---------|
| `pipeline/step2_manifolds/analyze_eigenvalues.py` | Scans REMA matrices across layers/sources, computes cumulative explained variance (k_80/k_90/k_95/k_99). Used to determine truncation rank via "kneedle" algorithm. |

### Curvature Analysis (Manifold Flattening Theory)
| Script | Purpose |
|--------|---------|
| `pipeline/step2_manifolds/analyze_curvature_32.py` | Generic curvature analysis tool. Measures how "flat" the representation manifold is at different layers. |
| `pipeline/step2_manifolds/analyze_curvature_MATH.py` | MATH-specific curvature analysis for Section 3.2's manifold flattening justification. |

---

## Section 3.3 — Geometric Conflict Verification

**Goal**: Quantify the structural disparity between syntactic and logical manifolds.

| Script | Purpose | What it Measures |
|--------|---------|-----------------|
| `pipeline/step3_verification/orthogonality_test.py` | **Bidirectional orthogonality scan**. Exp A: Fix REMA at elbow k, sweep MOM2 dimensions 1→512. Exp B: Fix MOM2 at k=256, sweep REMA dimensions. Maps REMA vectors through MLP to MOM2 input space for direct subspace comparison. | MaxCos (max cosine similarity), AvgLeak (Frobenius norm overlap) |
| `pipeline/step3_verification/spectrum_blimp_mom2.py` | Loads MOM2 C_syn, performs eigen-decomposition, projects BLiMP syntax activations onto each eigenvector. Measures syntactic projection energy per eigen-direction. | `analysis_results/layer_mom2_{model}_{layer}_spectral_data.pt` |
| `pipeline/step3_verification/spectrum_blimp_rema.py` | Mirror of above but uses REMA eigenvectors. Projects BLiMP syntax activations onto REMA's `projection_matrix` to show grammar does NOT align with REMA directions. | `analysis_results/layer_rema_{layer}_spectral_data_{dataset}.pt` |
| `pipeline/step3_verification/subspace_compare.py` | Computes principal angles (via SVD of U_a^T @ U_b) between pairs of REMA manifolds: Lite-GSM, Lite-Math, GSM-Math, Lite-Noise. | `radar_chart_data_full.pt` |
| `pipeline/step3_verification/subspace_overlap.py` | Computes 2D grid of normalized subspace overlap between MATH and GSM8K REMA manifolds: ||U_m^T @ U_g||_F^2 / min(k_m, k_g). | `rema_overlap_grid.pt` |

**Key metrics in paper**:
- Cumulative Spectral Energy (CSE): $\text{CSE}(k) = \sum_{i=1}^{k} \lambda_i / \operatorname{Tr}(\mathbf{C})$
- Syntactic Entanglement Coefficient: $\rho(\mathbf{C}, \mathcal{D}_{\text{BLiMP}})$ — Pearson correlation between spectral energy and BLiMP grammar energy in log-space
- Leakage Ratio: $\mathcal{L}_{ratio} = \|\mathbf{K}_{syn}^\top \mathbf{K}_{log}\|_F^2 / k_{log}$

---

## Section 4 — IntIso Method

**Goal**: Disentangle logical values from syntactic keys via the cross-manifold operator M = V̂_log K̂_syn^T.

| Script | Purpose | Key Mechanism |
|--------|---------|---------------|
| `pipeline/step4_intiso/ablation.py` | **Main experiment (64KB)**. Constructs the cross-manifold operator, runs ablation (α<0) for GSM8K and injection (α>0) for MATH, sweeps ranks and layers, evaluates Acc + weighted n-gram entropy H_w. | `get_hybrid_matrix_with_cache()` constructs M = U_top @ V_top^T; `reconstruct_matrix_from_source()` for SVD-based matrix control; `eval_generation_metrics()` with ChatML/Llama template detection |

**Experiment modes** (controlled by flags):
- `--run_gsm8k_sweep`: GSM8K REMA rank sweep (fix k_syn=128, vary k_log)
- `--run_math_sweep`: MATH REMA rank sweep
- `--run_mom2_sweep`: MOM2 rank sweep (fix k_log, vary k_syn)
- `--run_layer_sweep`: Layer 6/16/26 comparison
- `--run_ablation`: Ablation study (RandCov/RandVec) to verify both components are necessary
- `--run_capture`: Hidden-state trajectory capture for t-SNE visualization

**Matrix types compared**:
- **Hybrid** (IntIso): M = V_log @ K_syn^T (our method)
- **GenCov** (baseline): Standard covariance-aligned perturbation
- **Random**, **MOM2_Rand**, **MOM2_Self**, **REMA**: Control conditions
- **Ablation_RandCov**, **Ablation_RandVec**: Ablation controls

---

## Section 5 — Validation

| Script | Purpose |
|--------|---------|
| `pipeline/step5_validation/plot_analysis.py` | Generates dual-axis plot: MOM2 eigenvalue spectrum (log, red) + BLiMP grammar projection energy (blue). Shows syntax concentrates in top principal components. |
| `pipeline/step5_validation/draw.py` | Additional drawing/plotting for paper figures. |
| `pipeline/step5_validation/few_shot.py` | Few-shot diagnostic probing with GSM8K/MATH/MQuAKE prompts for qualitative analysis. |

**Metrics in paper**:
- **Exact Match Accuracy** on GSM8K and MATH-500 test sets
- **Weighted N-gram Entropy (H_w)**: aggregated Shannon entropy across n-grams (n ∈ [1,4]) to detect structural collapse

---

## Shared Utilities

| File | Purpose | Used By |
|------|---------|---------|
| `utils/nethook.py` | Model instrumentation (Trace, TraceDict, set_requires_grad) | All pipeline scripts |
| `utils/hparams.py` | Hyperparameter dataclass and JSON loading | build_c_syn, ablation |
| `utils/globals.py` | Global config (paths, device settings) | Various |
| `utils/generate.py` | Text generation with model-specific templates | ablation, few_shot |
| `utils/runningstats.py` | Running statistics (SecondMoment, Mean, CombinedStat) | build_c_syn |
| `utils/tok_dataset.py` | Tokenized dataset with masking utilities | build_c_syn |
| `utils/layer_stats_mom2.py` | Simplified MOM2 computation (ROME-compatible signature) | orthogonality_test, spectrum_blimp_mom2 |
| `utils/logit_lens.py` | LogitLens for interpretability | generate |
| `utils/perplexity.py` | Perplexity computation | (optional) |

## Data Loaders

| File | Dataset | Used By |
|------|---------|---------|
| `data/gsm8k.py` | GSM8K (grade school math, verbose reasoning) | build_c_log, ablation |
| `data/math.py` | MATH-500 (symbolic/competition math) | build_c_log, ablation |
| `data/mquake.py` | MQuAKE-CF (multi-hop counterfactual reasoning) | measure_sensitivity, gradient_probe |

## Shell Scripts (SLURM)

Each pipeline step directory contains `.sh` files with exact SLURM configurations used on the cluster:
- `gradient_probe.sh` — Causal tracing submission
- `rema.sh` — REMA manifold construction sweep (layers, sources, ranks)
- `rema_eigen.sh` — Eigenvalue analysis submission
- `orth_test.sh` — Orthogonality test submission
- `analyze_spectrum.sh` — BLiMP + MOM2 spectrum analysis
- `rema_compare.sh`, `rema_overlap.sh` — Subspace comparison submissions
- `ablation.sh` — IntIso main experiment with all sweep configs
- `rect_llama.sh`, `rema_memit.sh` — Additional experiment variants
- `basic.sh` — Basic evaluation submission

These scripts assume a SLURM cluster with conda environment `yanrongen_anyedit` and GPU partition.
