# BlueDot TAIS Puzzle #1 — solution

My solution + analysis for [BlueDot Impact's first Technical AI Safety Puzzle](https://github.com/SamDower/bluedot-tais-puzzle).

The puzzle: a small MLP head is trained on top of frozen MiniLM embeddings to predict 8 binary features from short text inputs. Seven features are encoded linearly at a marked hidden layer. One isn't. Figure out which feature, how it's actually encoded, then train a model with an even weirder representation.

## Headline results

**The non-linear feature is `country`.** At layer h2 (post-ReLU of hidden 2), a linear logistic-regression probe scores **46.9%** while an MLP probe scores **96.6%** — a 50-percentage-point gap. Every other feature has a probe gap below 0.5%.

**How it's encoded.** Country=1 and country=0 examples at h2:

- share the same mean (‖μ₊ − μ₋‖ = 0.013 vs activation norms ≈ 4.8)
- share the same 8-D subspace (top-8 principal angles all ≈ 0°)
- differ in **selective covariance shrinkage** along 2 specific axes (PC4 and PC6 of cov(F=1), with 10× spread ratios)

A quadratic discriminant on h2 recovers **96.7% — matching the MLP probe ceiling**. The next layer restores linearity by computing perpendicular distance via paired ReLUs (W₄ has 6 antipodal row pairs).

**Standard sparse autoencoders cannot decompose this feature.** Top-1 SAE feature = 50.3% (chance), top-64 = 64.1%, full 256-D probe = 92.1%. Country at h2 is structurally distributed — *not* a sparse linear feature. Same failure mode as the Engels et al. 2024 circular features, generalized to covariance-shape features.

**A class-aware quadratic feature recovers it from one scalar.** The log-likelihood ratio between the two same-mean Gaussians gives 96.6% from a single number.

**The encoding is engineered.** Three fresh retrainings of the puzzle's architecture all produce *linear* country (≥98.9%, μ_diff ≈ 6–8). The puzzle's geometry doesn't come out of standard training — adding a 3-line mean-cancellation regularizer reproduces μ_diff = 0.013 within numerical precision.

**For Task 3, three weirder models.** Each bottlenecks the target layer to 2-D and forces activations onto the unit circle at sub-class-specific angles; binary target = sign(cos(N·θ)) requires the N-th Fourier harmonic to decode. Models with periods 2, 3, 4. Linear probes below chance; only the right Fourier harmonic decodes. Causal rotations confirm the period structure exactly.

Full writeup: [SUBMISSION.docx](SUBMISSION.docx).

## Repository structure

```
analysis/
  common.py                    # load model, extract activations, shared utilities
  phase1_probes.py             # Task 1: linear vs MLP probes per feature/layer
  phase2_geometry.py           # initial hypothesis sweep
  phase2b_country.py           # per-country tagging
  phase2c_mechanism.py         # core Task 2 analysis: QDA, manifold, weight inspection
  phase2d_causal.py            # causal interventions on h2
  phase3v3_train.py            # Task 3: Models A and B (period 2 and 3)
  phase3v3_visualize.py        # Task 3 probes and figures
  phase5b_sae_and_gradient.py  # SAE on h2, gradient analysis
  phase5g_axis_ablation.py     # per-axis causal ablation
  phase6b_replicate_puzzle.py  # 3-seed replication
  phase6d_period4.py           # Task 3: Model D (period 4)
  phase7_reproduce_trick.py    # reproducing the engineering trick
  phase7d_quadratic_sae.py     # the log-likelihood-ratio feature
  phase8_lambda_sweep.py       # engineerability curve
  phase9_axis_meaning.py       # what v_manifold actually encodes
  phase10_decoder_stability.py # decoder direction stability across seeds
  phase10b_more_directions.py  # comparing 5 candidate axes across seeds
  ...                          # (~25 scripts total)

  figs/                        # all generated figures
  trained_models/              # model checkpoints
    replicas/                  # 3-seed replication models
```

## Reproducing the analysis

You'll need the puzzle's data and model checkpoint:

```bash
git clone https://github.com/SamDower/bluedot-tais-puzzle.git
cp -r bluedot-tais-puzzle/data .
cp bluedot-tais-puzzle/model.pt .
cp bluedot-tais-puzzle/feature_names.json .
```

Then install dependencies:

```bash
pip install sentence-transformers torch scikit-learn matplotlib
```

Run order (each script is standalone; common.py handles caching):

```bash
python analysis/common.py                 # extract & cache activations
python analysis/phase1_probes.py          # Task 1
python analysis/phase2c_mechanism.py      # Task 2 main result
python analysis/phase5g_axis_ablation.py  # circuit localization
python analysis/phase5b_sae_and_gradient.py
python analysis/phase7d_quadratic_sae.py
python analysis/phase7_reproduce_trick.py
python analysis/phase3v3_train.py         # Task 3 models A and B
python analysis/phase6d_period4.py        # Task 3 model D
python analysis/phase3v3_visualize.py
```

The full investigation runs end-to-end on CPU in about half an hour.

## Key figures

- `figs/axis_ablation.png` — per-axis causal ablation showing PC4/PC6 are the country circuit
- `figs/manifold_mechanism.png` — h2 in the (country axis, perpendicular) plane
- `figs/sae_comparison.png` — SAE variants vs hand-built quadratic features
- `figs/v_manifold_meaning.png` — what v_manifold actually encodes (template structure)
- `figs/model_A_v3_bottleneck_geometry.png` — Task 3 Model A circle (period 2)
- `figs/model_D_period4_geometry.png` — Task 3 Model D circle (period 4)
- `figs/side_by_side_puzzle_vs_modelA.png` — puzzle vs my model side-by-side

## Going public

This repo is private until **2026-06-13**, the day after BlueDot's "do not share publicly before 12th June" deadline. After that date, run any of:

```bash
# Bash / Mac / Linux
bash scripts/make_public.sh

# PowerShell / Windows
pwsh scripts/make_public.ps1

# Or directly via gh CLI
gh repo edit viraajminhas/bluedot-tais-puzzle-1-solution \
    --visibility public --accept-visibility-change-consequences
```

There's also a GitHub Action at `.github/workflows/auto-publish.yml` that fires daily and flips visibility automatically on or after 2026-06-13 — but it needs a personal-access-token secret (`GH_ADMIN_PAT`) since the default `GITHUB_TOKEN` cannot change repo visibility. Setup instructions inside the workflow file.

## License

MIT (see [LICENSE](LICENSE)).
