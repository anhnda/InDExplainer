# PyramidExplainer — Method Note

A hierarchical, interaction-aware, on-manifold attribution method for 2D image inputs. Model-agnostic (queries the predictor as an oracle). Written as a working note: the core construction, the one clean identity it rests on, and the limitations that must be stated rather than hidden.

---

## 1. Motivation

Additive attribution methods (Integrated Gradients, LIME) decompose a prediction into per-unit scalar contributions. Two consequences for vision:

1. **Re-composition is off-manifold.** Selecting the top-k% pixels (insertion test) produces a scattered, high-frequency input that lies off the data manifold. The model cannot recognize it — not because the wrong pixels were chosen, but because the recomposed input violates the data distribution. Even top-30% features can fail to recover the prediction.
2. **Interaction is invisible.** An image pattern is *local contrast structure* — a pixel is meaningful relative to its neighbors. IG is additive along a path; LIME fits a *linear* surrogate. Both are additive by construction and cannot represent "region A matters only next to region B." Raising the interaction order (2nd, 3rd, …) is both intractable (O(nᵏ)) and uninterpretable.

PyramidExplainer addresses both by (a) changing the unit from a pixel to a coherent region, (b) evaluating value on-manifold, and (c) capturing arbitrary-order interaction as the **non-additivity of a region's value relative to its sub-regions**, organized by a scale hierarchy so cost stays ~linear.

---

## 2. Setup

- Predictor `f : 𝒳 → ℝ`, treated as a black-box oracle (no gradients, no internals).
- Input `x ∈ ℝ^{H×W}`, baseline `x₀`.
- Data manifold `M`, assumed to have **local clique structure** (an MRF over the pixel grid): an in-distribution pixel value depends only on its neighborhood `N(i)`. This is an assumption on the *data*, not on `f`.

### Manifold-projection operator Φ

`Φ_R(x)` reveals region `R` and completes the complement so the result stays (approximately) on `M`:

- **Blur completion** — low-pass the complement. Cheap; removes local contrast outside `R` while keeping band-limited statistics. *Caveat: a half-sharp/half-blurred image is itself somewhat OOD at the mask boundary — this reduces OOD-ness, it does not fully reach M.*
- **Conditional inpainting** — fill the complement with `E[x_{Rᶜ} | x_R]` under the MRF / a learned generator. Genuinely targets `M`, but introduces a *second model* whose errors confound the attribution.

The explanation is defined **relative to the chosen Φ** (reference dependence, inherited from the baseline-choice problem in IG).

---

## 3. Construction

### Region tree

Build an agglomerative tree `T` over the image: leaves = small superpixels, internal nodes = merged regions, root = full image.

### Holistic on-manifold value

For any node (region) `R`:

```
v(R) = f(Φ_R(x)) − f(x₀)
```

`v(R)` is the value of the region *as a whole*. It contains interaction of **all orders** among `R`'s constituent pixels — no order-k enumeration is ever performed.

### Node synergy (whole minus parts)

For internal node `R` with children `c₁ … c_m`:

```
Δ(R) = v(R) − Σⱼ v(cⱼ)
```

- `Δ(R) > 0` → the sub-regions **cooperate** (a contrast/co-occurrence pattern that exists only when the pieces are together).
- `Δ(R) < 0` → **redundancy** (the pieces each already carried the signal).

`Δ(R)` is a single scalar per merge that absorbs arbitrary-order interaction among `R`'s descendants.

---

## 4. The core identity (completeness analog)

Summing synergy over all internal nodes telescopes:

```
v(root) = Σ_{leaves} v(leaf) + Σ_{internal} Δ(R)
```

**Proof sketch.** Every non-root node appears once as `+v(R)` (its own term) and once as `−v(R)` (inside its parent's sum). These cancel pairwise up the tree, leaving root minus leaves.

Interpretation: the whole-image effect decomposes into an **additive leaf part** (IG-like) plus **interaction collected at every merge scale**. This is the interaction-surviving analog of IG's completeness axiom — the guarantee that grouped/overlapping IG loses.

**Two dependence caveats (state, don't hide):**

1. **Φ-dependence.** The identity is algebraically exact for any `f`, `Φ`. But the *interpretation* of leaf terms is only as trustworthy as Φ at the **finest** scale — exactly where Φ is weakest (a single-superpixel reveal flirts with the scatter/OOD problem again).
2. **Tree-dependence.** The total `Σ Δ` is tree-invariant (it always equals `v(root) − Σ v(leaf)`), but **how synergy is distributed across scales is not**. "Synergy at scale s" is defined relative to the segmentation tree, not canonical.

---

## 5. Why not just higher-order interaction indices?

A fixed-order Shapley/Taylor expansion has O(nᵏ) terms at order k — intractable and uninterpretable beyond k=2. PyramidExplainer instead lets interaction live *inside* a region's holistic value and bounds the number of reported interaction terms to **O(number of tree nodes) ≈ O(n)**. Unbounded interaction order, linear-ish cost.

Related tractable alternatives (same philosophy, different structure):
- **Owen / coalition-structure Shapley** over `m` semantic groups: between-group interaction of any order, cost O(2^m) with `m` small. Use when you have ~5–15 regions and care about between-group structure.
- **Synergy-greedy set search**: directly find one minimal region `G` maximizing `synergy(G) = v(G) − Σ_{i∈G} v({i})` subject to sufficiency. Use when you just want *the* pattern.

---

## 6. On-manifold sufficiency criterion

Replaces additive completeness as the evaluation target:

> Region set `S` is **(ε, Φ)-sufficient** for `x` under `f` if `|f(Φ_S(x)) − f(x)| ≤ ε`, with `Φ_S(x) ∈ M`. Attribution = minimize `|S|` subject to this.

**Pattern-valued explanations (scoped proposition).** *Under the additional assumption that the discriminative signal has spatial extent exceeding single-clique scale*, minimal (ε, Φ)-sufficient sets are neighborhood-connected (pattern-shaped), not scattered.

> ⚠️ This proposition is **false without the spatial-extent condition.** A legitimately point-like discriminative feature (a specular highlight, a hot pixel) is in-distribution *and* outcome-relevant, but the MRF conditional smooths it away — so it is isolated yet not droppable. State the scope; do not claim it universally.

---

## 7. Honest comparison vs IG and LIME

| Axis | PyramidExplainer | IG | LIME |
|---|---|---|---|
| On-manifold insertion/deletion | **wins** (coherent, in-dist reveals) | scattered → OOD | on/off masks → OOD |
| Interaction sensitivity | **wins** (Δ captures all orders) | additive — blind | linear surrogate — blind |
| Standard zero-mask insertion | toss-up (metric favors baselines) | tuned for it | tuned for it |
| Localization (pointing/IoU) | strong | weak | medium |
| Cost | high (≥ LIME; worse if inpainter is heavy) | **cheap** (one path) | expensive |
| Gradient-faithfulness to f | partial (explains f under Φ) | **exact** (integrates f's gradients) | surrogate fit |
| Model-randomization sanity | **untested — risk** (Φ prior may leak) | mixed | mixed |

**Circularity trap.** The strongest win (on-manifold sufficiency) is on a metric the method is *built to optimize* — nearly tautological, and reviewers will catch it. Credibility requires winning on metrics **not** designed into the method:

1. **Synthetic interaction ground-truth** (class defined by two-region co-occurrence) scored by IoU — where additivity *provably* fails; the decisive, non-circular experiment.
2. **Independent faithfulness** — model-randomization sanity check; a held-out removal operator different from Φ.
3. **Standard zero-mask insertion/deletion** — accept a possible tie; tying on the baseline-favoring metric while winning on interaction is the strong combined story.

**Verdict.** Defensible claim is *not* "beats IG/LIME" but:

> *IG and LIME share an additivity assumption that provably fails for co-occurrence/contrast patterns. PyramidExplainer is the minimal method that captures arbitrary-order interaction while retaining a completeness-style identity, at the cost of reference dependence on (Φ, tree).*

---

## 8. Known limitations

- **Reference dependence (twice):** results depend on Φ and on the tree; only the leaf-vs-root split is tree-invariant.
- **Φ at fine scale:** leaf-value reliability degrades where Φ is weakest; the whole decomposition inherits this.
- **Locality of M ≠ locality of f:** ViTs/CNNs have large/global receptive fields. The MRF assumption is on the data manifold; it does **not** bound `f`'s interaction range. Tree-structured synergy can miss long-range interactions if the tree isn't aligned with the model's interaction structure.
- **Sanity-check risk:** the data prior in Φ (especially a learned inpainter) may produce plausible regions even for a randomized model. **Must be tested before any superiority claim.**
- **Cost:** value query per node; heavy if Φ is a diffusion/inpainting model.

---

## 9. Decisive validation experiments

1. **Synthetic co-occurrence dataset** — label depends on two regions appearing together; build the tree so they merge at one node; verify `Δ` spikes there and is ≈0 elsewhere. If `Δ` smears across scales, the segmentation must be **model-informed**, not purely image-based.
2. **Point-like feature test** — a known single-pixel discriminative feature; check whether minimal-`S` recovers it. If dropped, the §6 spatial-extent caveat is confirmed necessary.
3. **Φ-sensitivity** — same image, blur-Φ vs inpaint-Φ; measure how much minimal-`S` changes (quantifies reference dependence).
4. **Model-randomization** — run on a randomized network; the method should *not* produce confident coherent explanations.

---

## 10. One-line summary

Raise the unit, not the interaction order: assign coherent regions a holistic on-manifold value, read interaction as each region's synergy over its sub-regions, and recover a completeness-style identity by telescoping the synergy up a scale hierarchy — accepting that the result is defined relative to the chosen completion operator and segmentation tree.