# BlueDot TAIS Puzzle #1

## What I'm answering

This puzzle gave us a small neural net that predicts 8 binary features from short text inputs. Seven of the features are encoded linearly at the marked layer. One isn't. The job is to find that feature, explain how it's actually encoded, and then train your own model that does something even weirder.

I picked this because I'd just read the Engels et al. paper on non-linear features in language models and wanted to actually do the kind of analysis they describe instead of just reading about it. Turned out to be way deeper than I expected.

**The non-linear feature is `country`.** The encoding is two same-mean Gaussians with different covariance shapes, which is invisible to every standard interpretability tool (linear probes, SAEs, gradient analysis) but recoverable from a single class-aware quadratic feature. I also showed the encoding is engineered rather than emergent: 3 fresh retrainings of the same architecture all produce clean linear country instead.

## Task 1: finding F

I trained two probes on each feature at each hidden layer: a logistic regression (linear) and a small MLP (can pick up nonlinear stuff). If a feature is encoded linearly, both probes match. If a feature is encoded nonlinearly, the MLP probe beats the linear probe by a lot.

Results at h2 (the layer the puzzle marks):

| feature | linear probe | MLP probe | gap |
|---|---|---|---|
| **country** | **0.469** | **0.966** | **+0.497** |
| number | 0.975 | 0.978 | +0.003 |
| sentiment | 0.981 | 0.982 | +0.001 |
| color | 0.973 | 0.973 | 0.000 |
| food, question, person, body_part | ≥ 0.98 | same | 0.000 |

Country has a 50-point gap. Every other feature has a gap below 0.5%.

Across layers, country goes from 99.4% linear at the embedding → 99.3% at h1 → **46.9% at h2** → 96.2% at h3. The signal is linearly readable everywhere except at h2 specifically, where it gets actively broken and then reconstructed one layer later. No other feature does this.

The linear probe scoring 46.9% (below chance) is also a tell. With balanced classes a random classifier should get exactly 50%; scoring below that means the linear probe is being actively misled, not just failing to find a direction. This only happens when the class means coincide.

## Task 2: how country is encoded

I tried the obvious hypotheses first. Country isn't a radial/circular encoding, isn't an XOR with another feature, and k-means on country=1 doesn't reveal clean sub-clusters. The thing that finally worked was comparing class covariances.

### Three numbers that explain everything

1. **‖μ_country=1 − μ_country=0‖ = 0.013** at h2, against activation norms of ~4.8. The two class means coincide to numerical precision, which is why any linear probe fails.

2. **Both classes live in the same 8-D subspace.** Principal angles between the top-8 eigenvectors of each class covariance: 0.0°, 0.0°, 0.0°, 0.0°, 0.0°, 0.3°, 0.5°, 36.1°. So country=1 isn't in a different region of h2, it's in the same place as country=0, just shaped differently.

3. **Country=1 is selectively compressed along 2 specific axes.** Looking at the ratio of variances along each of country=1's principal axes, PC4 and PC6 have country=0 spreading ~10× wider than country=1. Other axes are roughly equal.

Per-axis causal ablation confirms PC4 and PC6 are the country circuit: zeroing either drops country accuracy by ~20 points; zeroing the top 4 high-ratio axes drops country to 49.7% (chance). PC5 turns out to be the question circuit, PCs 0–3 carry the other features.

![Per-axis ablation: PC4 and PC6 are country (red), PC5 is question, PCs 0–3 carry number/color/person/body_part.](figs/axis_ablation.png)

### A quadratic discriminant matches the MLP probe exactly

If country is encoded as two same-mean Gaussians with different covariances, then QDA (which models exactly this) should work. It does: **96.7% test accuracy, within 0.1% of the MLP probe.** The two-Gaussian picture explains everything.

A single nonlinear feature I built — the perpendicular distance from each example to country=1's top eigenvector — gets 89.5% from one scalar. Most of the country signal lives in one direction perpendicular to the country manifold.

![h2 in the (country axis, perpendicular) plane. Country=1 (red) forms a thin strip; country=0 (grey) spreads broadly. Right: histogram of perpendicular distance per class.](figs/manifold_mechanism.png)

### How the next layer recovers linearity

The next linear layer (h2 → pre-h3) has 6 pairs of weight rows with cosine similarity ≤ −0.8. This is the standard ReLU trick: `ReLU(+w·x) + ReLU(−w·x) = |w·x|`. The model is computing absolute value of perpendicular projections, giving a positive scalar at h3 that's monotone in off-manifold distance and linearly readable.

### Causal interventions

Editing h2 directly on the test set and re-running through downstream layers:

| edit | F=1 country logit | F=0 country logit | F=0 accuracy |
|---|---|---|---|
| baseline | +7.94 | −10.74 | 93.5% |
| project F=0 onto country subspace | +10.88 | **+10.28** | **0.0%** |
| amplify F=0 perpendicular component 2× | unchanged | **−36.95** | 99.2% |

The country logit causally tracks perpendicular distance to the country manifold. Forcing country=0 examples onto the manifold makes the model fully believe they're country=1.

### Standard sparse autoencoders cannot find country

I trained a top-k=8 SAE on h2 (the modern interpretability default). Reconstruction is essentially perfect (test MSE 0.0043), but country is invisible to single SAE features:

| # of top country-selective SAE features used | test accuracy |
|---|---|
| top 1 | **50.3% (chance)** |
| top 64 | 64.1% |
| all 256 | 92.1% |

The full 256-D probe works because the SAE preserves h2's geometry, but no subset of country-aligned features captures the feature. Country is structurally distributed, not concentrated in any sparse direction. This is the Engels-paper "non-LRH feature" failure mode generalized to covariance-shape features.

### A single class-aware quadratic feature recovers it at the MLP ceiling

If the encoding is two same-mean Gaussians, the optimal one-feature detector is the log-likelihood ratio between them:

```
LLR(x) = (x - μ)ᵀ Σ_F=0⁻¹ (x - μ) - (x - μ)ᵀ Σ_F=1⁻¹ (x - μ)
```

Trained as a 1-feature logistic regression: **test accuracy = 96.6%**, matching the MLP probe exactly. So country is decodable from one scalar, but only if you know to use this specific quadratic form. SAEs can't learn this because the right whitening direction depends on class labels.

![Country decoding from h2 via different feature classes. Hand-built quadratic features (green) match the MLP ceiling; every learned SAE top-1 sits at chance (red).](figs/sae_comparison.png)

### The encoding is engineered, not generic

This was the result I didn't expect. I trained 3 fresh copies of the puzzle's exact architecture from scratch with different random seeds.

| | puzzle | seed 101 | seed 202 | seed 303 |
|---|---|---|---|---|
| country linear probe at h2 | **46.9%** | 99.3% | 99.0% | 98.9% |
| ‖μ_diff‖ at h2 | **0.013** | 6.42 | 7.85 | 6.11 |

None of the 3 naive replicas reproduces the geometry. Standard training gives clean linear country every time. The puzzle has to be engineered.

I tried to reverse-engineer the trick. The simplest version that works is a 3-line regularizer that pushes the class means together:

```python
mu_p = h2[y == 1].mean(0)
mu_n = h2[y == 0].mean(0)
loss = bce(logits, labels) + 10.0 * ((mu_p - mu_n) ** 2).sum()
```

After 100 epochs my reproduction has ‖μ_diff‖ = 0.009 (puzzle: 0.013). The catch is that at this strength the model also stops being able to predict country at all — MLP probe and output both collapse to chance. The puzzle keeps both linear-unreadability AND nonlinear-readability, which my simple regularizer doesn't. So the actual engineering is more delicate than mine, probably a much smaller λ with way longer training.

### What the country axis actually means

The top eigenvector of cov(country=1), which I'd been calling "the country axis," doesn't correlate with anything obvious about country names. MiniLM embeddings of bare names get r ≈ 0, training frequency r = 0.06, alphabet position r = 0.04, gradient of country logit cos = 0.013.

The reason none of those work is that **v_manifold isn't actually the country decoder direction**. I traced the country output direction backward through the model (W₅[country] through W₄'s ReLU back to h2) and found the resulting "decoder direction" has only |cos| = 0.18 with v_manifold. The actual decoder lives perpendicular to v_manifold, in the PC4/PC6 subspace.

So what does v_manifold encode? Per-country correlation of v_manifold projection with co-occurrence of other features:

| co-occurring feature | correlation |
|---|---|
| **person** | **−0.65** |
| **body_part** | **−0.42** |
| **number** | **+0.37** |
| color | +0.24 |

Countries that show up in texts with person names and body parts project low; countries that show up with numbers and colors project high. The extreme texts confirm it:

- High v_manifold: "The musician is a fan of the incredible clothes painted gold from Taiwan in 12 minutes."
- Low v_manifold: "Does Peter pack the outstanding pancakes in Norway covering the eyelash?"

v_manifold isn't the country axis. It's the **template-structure axis** — variance within country=1 examples that comes from incidental differences in which templates each country tends to appear in.

I also checked which direction (if any) IS stable across my 3 seed replicas:

| candidate direction | mean \|cos\| across 4 models |
|---|---|
| v_manifold (PC1 of cov(F=1)) | **0.39** |
| all-data PC1 | 0.37 |
| country decoder direction | 0.08 |
| LDA direction (mean F=1 − mean F=0) | 0.07 |
| linear-probe weight | **0.006** (random) |

Only the data-determined directions (v_manifold and all-data PC1) are stable across seeds. Every country-task-specific direction is essentially random across runs. So **there is no canonical country direction in this model's hidden space** — each training run uses a different geometric arrangement to predict country. Only template structure is partially shared, because it lives in the data.

![What v_manifold actually encodes: per-feature co-occurrence correlations (left), per-template projection distributions (middle), extreme texts (right).](figs/v_manifold_meaning.png)

## Task 3: three weirder models

The puzzle's encoding is non-linear, but it's decodable from a single nonlinear feature (perp distance, 89.5%). I wanted to train models that are strictly harder to read.

My approach: force activations onto a 2D circle and make the binary label depend on a periodic function of the angle. If I put 6 sub-classes at 6 evenly spaced angles around the unit circle with labels alternating 0,1,0,1,0,1, then decoding requires the 3rd Fourier harmonic. For period N > 1, no linear function of (cos θ, sin θ) can match sign(cos(Nθ)). So linear probes are provably below chance.

Architecture: same MLP head as the puzzle but with L bottlenecked to 2D and an auxiliary loss forcing activations onto the unit circle at sub-class-specific angles. Three models with periods 2, 3, and 4.

| | Model A (period 2) | Model B (period 3) | Model D (period 4) |
|---|---|---|---|
| sub-classes | 4 from (color, food) Gray-coded | 6 from template_id mod 6 | 8 from template_id |
| test accuracy | 97.5% | 93.7% | 94.1% |
| **linear probe on L** | **30.3%** | **37.6%** | **47.9%** |
| Fourier k=N (right harmonic) | **97.6%** | **94.4%** | **94.3%** |
| Fourier wrong harmonics | 30–50% | 38–56% | 49–55% |

Every model's linear probe is below chance. Only the right Fourier harmonic decodes the target. Wrong harmonics fail by 30–50 percentage points.

![Model A's 2D bottleneck: 4 clusters at 0°/90°/180°/270°, target alternates each cluster.](figs/model_A_v3_bottleneck_geometry.png)

### Causal rotations confirm the period structure

For each model, rotating L by various angles and re-running downstream:

- **Model A (period 2):** 90° rotation → 99.9% flipped. **180° rotation → 0.0% flipped** (period 2: antipodal point has same label).
- **Model B (period 3):** 60°/180°/300° → ~97% flipped. **120°/240° → 1–2% flipped** (period 3: 2-arc shifts preserve target).
- **Model D (period 4):** 45°/135°/225°/315° → ~96% flipped. **90°/180°/270° → 0–3% flipped**.

Rotation = π (mod 2π/N) flips the prediction; rotation ≡ 0 (mod 2π/N) preserves it. The downstream layers are physically computing sign(cos(Nθ)).

![Side-by-side: puzzle's country geometry (left, distributed manifold) vs Model A's circle (right, 4 crisp clusters). Both non-linear, in different ways.](figs/side_by_side_puzzle_vs_modelA.png)

### Things that didn't work

A few false starts I tried first:

- **Aux loss on raw post-ReLU h2 coords.** ReLU clips negatives, so only the sub-class at angle 0° could satisfy `(cos θ, sin θ)`. The other 3 sub-classes collapsed to the origin and the "circle" was decorative — the model used other h2 dimensions to predict the target. Fixed by using a learned linear projection of h2 instead.
- **Wrong sub-class to angle mapping.** Originally used `sub = color*2 + food`, which puts target=1 at angles 90° and 180°, linearly separable by `sin θ − cos θ > 0`. Gray-coded the assignment to fix.
- **Möbius capstone.** Tried a 3D topologically twisted encoding where the binary depends on "side" of the strip. Trained to 97.6% but linear probe on the (x,y) projection alone was also 97.6% — my sub-class positions in (x,y) happened to be linearly separable. Möbius topology only obstructs path-dependent decoding, not static-point classification, so this approach was kind of doomed.
- **4-way XOR target** (color ⊕ food ⊕ sentiment ⊕ body_part). Train acc reached 99% but test stayed at 53% (chance). 4-way parity is memorisable on 7000 examples without learning the underlying detectors.

## What I think this means

The most interesting thing isn't the specific finding about country. It's that **the puzzle author engineered a feature that's invisible to every standard interpretability tool with what's basically a 3-line addition to the loss function.** Linear probes miss it. SAEs miss it. Gradient analysis misses it. LDA misses it. All give chance accuracy or weak signal, even though the feature is right there and the model uses it confidently.

The good news is the encoding has a clean fingerprint: same class means, different class covariances. If you check ‖μ_F=1 − μ_F=0‖ relative to the average activation norm and find it tiny, you're in this trap, and a quadratic discriminant or the LLR feature I built will catch what linear methods miss.

The bigger worry: this feature was engineered for a puzzle, but the same kind of geometry could emerge accidentally during training, or be created deliberately by someone trying to hide a backdoor. The standard interpretability stack wouldn't catch it.

## Code

All the analysis is in the `analysis/` directory: about 25 scripts that walk through the investigation step by step. The whole thing runs end-to-end on CPU in under 30 minutes. Trained model weights in `analysis/trained_models/`, figures in `analysis/figs/`.

The key scripts:
- `phase1_probes.py` — Task 1
- `phase2c_mechanism.py` — Task 2 main analysis
- `phase5g_axis_ablation.py` — per-axis circuit localization
- `phase5b_sae_and_gradient.py` — SAE failure
- `phase7d_quadratic_sae.py` — the LLR feature
- `phase7_reproduce_trick.py` — reproducing the engineering
- `phase3v3_train.py`, `phase6d_period4.py` — Task 3 models
- `phase9_axis_meaning.py`, `phase10_decoder_stability.py` — axis interpretation
