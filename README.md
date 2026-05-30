# In-Distribution Sufficiency Attribution (ResNet-50 / ImageNet-1k)

Three attribution methods behind one `Explainer` interface, sharing a
strong-blur self-reference `b = blur_sigma(x)`.

## Layout
- `backbone.py`        ResNet-50 IMAGENET1K_V2 loader + preprocessing/(de)normalization
- `explainers/base.py` `Explainer` interface, `AttributionResult`, blur reference
- `explainers/lime.py` LIME, grid superpixels (default 16x16), blur-out perturbations
- `explainers/ig.py`   Integrated Gradients with the blur reference as baseline
- `explainers/sufficiency.py`  the proposed method (min sufficient region)
- `explain.py`         main driver: one input image -> one output file per method
- `evaluate.py`        insertion/deletion curves (+AUC) and sigma sensitivity sweep

## Run
```
pip install torch torchvision matplotlib
python explain.py --image dog.jpg --out outputs            # all three methods
python explain.py --image dog.jpg --out outputs --target 207
python explain.py --image dog.jpg --methods sufficiency --stochastic
```
Outputs: `outputs/<method>.png` (input | blur ref | heatmap | overlay) + `summary.txt`.

## Evaluate
```
python evaluate.py curves --image dog.jpg --out eval_out          # all methods
python evaluate.py curves --image dog.jpg --methods sufficiency ig
python evaluate.py sweep  --image dog.jpg --sigmas 5 8 11 15 21    # sufficiency only
python evaluate.py both   --image dog.jpg --out eval_out
```
- **curves**: ranks pixels by |attribution|, then insertion (blur->sharp, top-down)
  and deletion (sharp->blur, top-down) using the SAME blur reference as the
  background — so insertion is the exact inverse of the sufficiency composite.
  Writes `insertion_deletion.png` + `curves.json`. Higher insertion-AUC / lower
  deletion-AUC = more faithful. Sharp early insertion rise = compact evidence.
- **sweep**: for each sigma, reports f_b (neutrality), f_phi (sufficiency), mask
  mass, and insertion/deletion AUC. Writes `sigma_sweep.png` + `sweep.json`, and
  prints a WARNING for any sigma where `f_b >= f_x` (blur not class-neutral).
  Pick the sigma with low f_b, high f_phi, low mass, high insertion-AUC.

## Interface
```python
from xai_suff.backbone import load_backbone, load_image, get_class_names
from xai_suff.explainers import SufficiencyExplainer

model = load_backbone("cuda")
x = load_image("dog.jpg", "cuda")
exp = SufficiencyExplainer(model, target_class=None, device="cuda",
                           class_names=get_class_names())
res = exp.explain(x)           # res.attribution: (H,W) map; res.f_x/f_b/f_phi diagnostics
```

## The sufficiency method

**Core idea.** Importance = the *smallest sharp region that still triggers the
class when everything else is blurred away*. Instead of asking "which pixels does
the gradient point at", we directly search for the minimal patch of original
detail that, composited against a blurred version of the same image, keeps the
prediction near its original value. The optimized mask *is* the explanation.

**Composite.** A soft mask `m` in `[0,1]` blends sharp image and blurred self:

    Phi(x, m) = m * x + (1 - m) * blur_sigma(x)

Kept pixels stay sharp; the rest dissolve into the blur. The background is a
strong blur of `x` itself, so it carries real color statistics (on-manifold) but
no object high-frequency detail — unlike a black/gray/noise baseline, which
pushes the composite off-distribution and makes the classifier respond to the
artifact rather than the evidence.

**Optimization problem.** Smallest coherent sharp region subject to keeping the
prediction high:

    min_m  |m|_1 + lambda * TV(m)
    s.t.   f(Phi(x, m)) >= (1 - eps) * f(x)

  - `|m|_1`  — minimal mass: least sharp area needed.
  - `TV(m)`  — total variation: an organic, connected region rather than scattered
    speckle; combined with the feathered edge this kills shape/boundary leakage.
  - constraint — "how little of the object, before `f`, still fires". Satisfiable
    at small `m` iff the evidence is compact.

**Solver — Lagrangian + dual ascent.** The fidelity term is treated as a hard
constraint with a multiplier `mu >= 0`, so sparsity and faithfulness auto-balance
per image (no manual `lambda` sweep on the constraint):

    L(m, mu) = |m|_1 + lambda * TV(m) + mu * (target_recovery - recovery)

Descend `m` (Adam on the mask logits), ascend `mu` (clamped to `mu_max` to avoid
runaway). Per-instance, surrogate-free: only the frozen `f` is queried, no trained
explainer, no segmentation.

**Two design choices baked into the implementation:**

1. **`f(b)`-relative constraint.** "Recovery" is measured as the gap recovered
   from the blur floor `f(b)` up to `f(x)`, not implicitly relative to zero:
   `recovery = f(Phi) - f(b)`, and the mask must reach `(1 - eps) * (f(x) - f(b))`.
   This stops the mask getting credit for class signal the blur itself already
   provides. `f_b` is logged so blur neutrality is *audited, not assumed*.

2. **Feathered mask.** `m` is a sigmoid of Gaussian-blurred free logits, so its
   edges are soft. A hard mask boundary is itself a shape cue the classifier can
   exploit (boundary leak); feathering removes it.

**Output.** A continuous per-pixel sufficiency map `m*` in `[0,1]` (graduated, not
per-segment constants like LIME). Threshold it at increasing budgets for an
insertion curve (`evaluate.py`); a sharp early rise means compact evidence. The
deletion curve on `1 - m*` is the necessity counterpart and guards against
degenerate masks.

**Three properties it targets.**
- *In-distribution* — blurred background is on-manifold; soft mask removes the
  boundary edge → no flat-canvas/noise OOD, no shape leak.
- *Self-contained* — blur is parameter-light, only the frozen `f`; no surrogate,
  no segmentation, no trained explainer.
- *Per-pixel, graduated* — continuous `m`; top-k via thresholds yields a curve,
  not a single arbitrary cutoff.

**Known liability.** It is still mask-optimization against a single frozen `f`, so
a low-mass `m` can satisfy the constraint without being semantically faithful (the
adversarial-mask problem). TV + L1 + feathering suppress this; the `--stochastic`
option (expected recovery over Bernoulli-sampled hard masks, straight-through
gradient) is the stronger defense.

### Key parameters (`SufficiencyExplainer`)
- `sigma` (11.0) — blur strength of the reference; the one real knob, swept by
  `evaluate.py sweep`. Too low → blur still classifies (`f_b` high); too high →
  drifts toward an OOD flat field.
- `lam` (0.05) — TV weight (region coherence vs. sparsity).
- `eps` (0.10) — tolerance: recover at least `(1 - eps)` of the `f(x) - f(b)` gap.
- `steps` (300), `lr` (0.05) — mask optimization budget / step size.
- `rho` (0.5), `mu_init` (5.0), `mu_max` (50.0) — dual-ascent rate, init, clamp.
- `feather` (2.0) — Gaussian sigma applied to mask logits (edge softness).
- `stochastic` (False), `n_mc` (4) — expected-recovery mode and its MC samples.

## Notes & gotchas
- **Always check `f_b` in the output** — if `f_b >= f_x` the blur isn't
  class-neutral for this image and the map is not meaningful (lower `--sigma`, or
  the class is texture/color-driven and blur is the wrong neutral for it).
- `res.attribution` is `(H,W)` in `[0,1]`; `res.f_x / f_b / f_phi` and
  `res.extras` (gap, final_mass, mu_final, history) hold the diagnostics.
- Pretrained weights download from download.pytorch.org; ensure that host is
  reachable on your machine (it is blocked in some sandboxes).