"""PyramidExplainer -- hierarchical, interaction-aware, on-manifold attribution.

Implements the method note (PyramidExplainer.md) against the package contract:

    Explainer(model, target_class=None, device=...)
    explainer.explain(x) -> AttributionResult

Core construction
-----------------
1. Segment the image into superpixels (leaves).
2. Build an agglomerative region tree T as the GREEDY MINIMIZER of an explicit
   tree objective, NOT as an ad-hoc "merge most-similar pair" loop:

       J(T) = sum_{R in I} w(R) * |Delta(R)|,   w(R) = 1 / area(R)

   Rationale (P1 leaf honesty, P2 localization): every tree is exactly correct
   (the telescoping identity below holds for ANY partition tree), so the tree
   cannot change correctness -- it only chooses WHERE on the scale axis the
   (conserved) non-additivity mass appears. We want fine-scale merges additive
   (small |Delta|, weighted heavily by 1/area) and synergy concentrated at a few
   coarse nodes. Minimizing J pushes residual mass upward.

   Each merge creates exactly one internal node, so committing (u,v) adds a
   single term w(u|v)*|Delta(u|v)| to J. The steepest-descent step is therefore

       (u*,v*) = argmin_{(u,v) in adj} w(u|v) * | v(u|v) - v(u) - v(v) |

   This is Ward's method with a MODEL-DEFINED linkage: we merge the adjacent
   pair of least *explanation non-additivity* increase, evaluating f under Phi
   rather than comparing colors. The merge rule is the derivative of J, not an
   independent heuristic.

   We minimize (not maximize) |Delta|: a synergy that survives a tree built to
   defer it is robust evidence; one found by a tree built to seek it is an
   artifact. Minimization is what P1/P2 demand and what makes residuals trusted.

   Cost control -- "blind proposal, model-aware selection": a pure greedy would
   trial v(u|v) for every adjacent pair every level (O(nE) forward passes). We
   instead shortlist the k most-similar adjacent pairs by a *blind* signal
   (backbone intermediate-feature distance, falling back to mean color), then
   score only those k by the model-aware criterion. Chosen-merge values are
   cached and reused as tree nodes, so only the <= k-1 rejected trials cost
   extra passes => O(n*k) forward passes, a tunable constant over 2n-1.

3. Holistic on-manifold value of any region R (blur-completion Phi):

       Phi_R(x) = reveal R sharp, replace complement R^c with blur_sigma(x)
       v(R)     = f(Phi_R(x)) - f(x0)            with x0 = full blur reference b

4. Node synergy (whole minus parts) for internal node R with children c_1..c_m:

       Delta(R) = v(R) - sum_j v(c_j)

   Delta>0 cooperation, Delta<0 redundancy.

5. Completeness-style identity (telescoping), preserved because the tree stays a
   strict binary partition -- only WHICH coalitions are sampled changes:

       v(root) = sum_{leaves} v(leaf) + sum_{internal} Delta(R)

Per-pixel attribution
----------------------
The returned (H,W) map is the leaf-additive part: each pixel takes the value of
its leaf superpixel, v(leaf) / area(leaf). Interaction is reported separately in
`extras` (the full tree with each node's v and Delta).

Reference dependence (Phi, tree) is stated, not hidden: chosen sigma, segmentation
params, merge objective settings, and the value-query budget are logged in `extras`.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F

try:  # scikit-image is used for the leaf segmentation
    from skimage.segmentation import slic
    from skimage.future import graph as skgraph  # region adjacency graph
    _HAS_SKIMAGE = True
except Exception:  # pragma: no cover - skimage layout varies across versions
    try:
        from skimage.segmentation import slic
        from skimage.graph import rag_mean_color  # newer skimage location
        _HAS_SKIMAGE = True
        skgraph = None
    except Exception:
        _HAS_SKIMAGE = False
        skgraph = None

from .base import AttributionResult, Explainer, blur_reference, denormalize


@dataclass
class _Node:
    """One region in the agglomerative tree."""
    id: int
    mask: np.ndarray            # (H,W) bool, pixels belonging to this region
    children: list             # list[_Node]; empty for leaves
    v: float = 0.0             # holistic on-manifold value f(Phi_R(x)) - f(x0)
    delta: float = 0.0         # synergy v(R) - sum_j v(child_j); 0 for leaves

    @property
    def is_leaf(self) -> bool:
        return len(self.children) == 0

    @property
    def area(self) -> int:
        return int(self.mask.sum())


class PyramidExplainer(Explainer):
    name = "pyramid"

    def __init__(
        self,
        *args,
        sigma: float = 11.0,        # blur strength for Phi complement (on-manifold)
        n_segments: int = 144,       # target number of leaf superpixels (SLIC)
        compactness: float = 2,  # SLIC compactness
        max_nodes: Optional[int] = None,  # (unused now; kept for API compat)
        # --- merge-objective controls ----------------------------------- #
        shortlist_k: int = 6,        # blind-proposal shortlist size per merge step
        feature_layer: str = "layer3",  # backbone layer for blind feature distance
        merge_mode: str = "value",   # "value" (objective-driven) | "color" (legacy)
        weight_eps: float = 1.0,     # w(R) = 1 / (area(R) + weight_eps)
        **kw,
    ):
        super().__init__(*args, **kw)
        self.sigma = sigma
        self.n_segments = n_segments
        self.compactness = compactness
        self.max_nodes = max_nodes
        self.shortlist_k = shortlist_k
        self.feature_layer = feature_layer
        self.merge_mode = merge_mode
        self.weight_eps = weight_eps
        # filled during explain(); bookkeeping for the objective-driven build.
        self._build_queries = 0

    # ------------------------------------------------------------------ #
    # Phi: blur-completion manifold-projection operator
    # ------------------------------------------------------------------ #
    def _phi(self, x: torch.Tensor, b: torch.Tensor, mask_bool: np.ndarray) -> torch.Tensor:
        """Reveal `mask` sharp; complete the complement with blur reference b.

        Phi_R(x) = m * x + (1 - m) * b,  m the (H,W) {0,1} region indicator.
        """
        m = torch.as_tensor(mask_bool, dtype=x.dtype, device=x.device)
        m = m.view(1, 1, *mask_bool.shape)  # (1,1,H,W) broadcast over channels
        return m * x + (1.0 - m) * b

    def _target_prob(self, comp: torch.Tensor, target: int) -> float:
        with torch.no_grad():
            return float(F.softmax(self.model(comp), dim=1)[:, target].item())

    def _value(self, x, b, mask_bool, x0_val: float, target: int) -> float:
        """v(R) = f(Phi_R(x)) - f(x0), counting one forward pass."""
        self._build_queries += 1
        return self._target_prob(self._phi(x, b, mask_bool), target) - x0_val

    # ------------------------------------------------------------------ #
    # leaf segmentation
    # ------------------------------------------------------------------ #
    def _leaf_labels(self, img01: np.ndarray) -> np.ndarray:
        """Return an (H,W) int label map of leaf superpixels."""
        if _HAS_SKIMAGE:
            labels = slic(
                img01,
                n_segments=self.n_segments,
                compactness=self.compactness,
                start_label=0,
                channel_axis=2,
            )
            return labels.astype(np.int64)
        # Fallback: regular grid tiling if skimage is unavailable.
        H, W = img01.shape[:2]
        side = max(1, int(round(np.sqrt(self.n_segments))))
        ys = np.linspace(0, side, H, endpoint=False).astype(np.int64)
        xs = np.linspace(0, side, W, endpoint=False).astype(np.int64)
        labels = (ys[:, None] * side + xs[None, :]).astype(np.int64)
        return labels

    # ------------------------------------------------------------------ #
    # blind proposal signal: backbone intermediate-feature map (model-blind:
    # it never sees the target class or v; it just groups regions the network
    # represents similarly, a better scaffold than raw color).
    # ------------------------------------------------------------------ #
    def _feature_map(self, x: torch.Tensor, H: int, W: int) -> Optional[np.ndarray]:
        """Return an (H,W,C) per-pixel feature map from `feature_layer`, or None.

        Hooks the named module on the backbone, runs one forward pass on the
        sharp input, and upsamples the activation to (H,W). Falls back to None
        if the layer is not found, in which case the caller uses mean color.
        """
        module = dict(self.model.named_modules()).get(self.feature_layer)
        if module is None:
            return None
        feats = {}

        def hook(_m, _inp, out):
            feats["act"] = out.detach()

        handle = module.register_forward_hook(hook)
        try:
            with torch.no_grad():
                self.model(x)
        finally:
            handle.remove()
        act = feats.get("act")
        if act is None:
            return None
        # (1,C,h,w) -> (1,C,H,W) -> (H,W,C)
        act = F.interpolate(act, size=(H, W), mode="bilinear", align_corners=False)
        fmap = act[0].permute(1, 2, 0).cpu().numpy().astype(np.float64)
        return fmap

    @staticmethod
    def _region_descriptor(desc_field: np.ndarray, mask: np.ndarray) -> np.ndarray:
        """Mean of a per-pixel descriptor field (color or feature) over a region."""
        return desc_field[mask].mean(axis=0)

    # ------------------------------------------------------------------ #
    # agglomerative region tree -- greedy minimizer of J(T) = sum w(R)|Delta(R)|
    # ------------------------------------------------------------------ #
    def _build_tree(
        self,
        img01: np.ndarray,
        labels: np.ndarray,
        x: torch.Tensor,
        b: torch.Tensor,
        x0_val: float,
        target: int,
        desc_field: np.ndarray,
    ) -> _Node:
        """Build the tree by greedy descent on the merge objective.

        desc_field: (H,W,D) per-pixel BLIND descriptor used only to shortlist
        candidate merges (feature activations if available, else color). The
        actual merge choice is MODEL-AWARE: among the shortlist we pick the pair
        of least weighted residual w(u|v)*|v(u|v) - v(u) - v(v)|.
        """
        H, W = labels.shape
        uniq = np.unique(labels)

        # --- leaf nodes + their (cached) values v(leaf) ------------------- #
        next_id = 0
        nodes: dict[int, _Node] = {}
        for lab in uniq:
            mask = labels == lab
            node = _Node(id=next_id, mask=mask, children=[])
            node.v = self._value(x, b, mask, x0_val, target)  # leaf value, cached
            nodes[next_id] = node
            next_id += 1

        active = set(nodes.keys())

        # Blind descriptor per active region (for shortlisting only).
        desc = {nid: self._region_descriptor(desc_field, nodes[nid].mask) for nid in active}

        # --- adjacency among current active regions ---------------------- #
        def adjacency(masks: dict[int, np.ndarray]) -> set:
            owner = -np.ones((H, W), dtype=np.int64)
            for nid, m in masks.items():
                owner[m] = nid
            pairs = set()
            a, b_ = owner[:, :-1], owner[:, 1:]
            for u, v in zip(a[a != b_], b_[a != b_]):
                if u >= 0 and v >= 0:
                    pairs.add((min(int(u), int(v)), max(int(u), int(v))))
            a, b_ = owner[:-1, :], owner[1:, :]
            for u, v in zip(a[a != b_], b_[a != b_]):
                if u >= 0 and v >= 0:
                    pairs.add((min(int(u), int(v)), max(int(u), int(v))))
            return pairs

        masks = {nid: nodes[nid].mask for nid in active}
        adj = adjacency(masks)

        def blind_dist(p) -> float:
            return float(np.linalg.norm(desc[p[0]] - desc[p[1]]))

        def weight(area: int) -> float:
            return 1.0 / (float(area) + self.weight_eps)

        # Greedy descent on J: each step, propose the k blind-closest adjacent
        # pairs, then commit the one of least weighted residual (model-aware).
        while len(active) > 1:
            if not adj:
                # Disconnected remainder (essentially never with 4-conn SLIC):
                # merge the two spatially-closest actives by centroid -- a mild,
                # principled fallback rather than an arbitrary pick.
                acts = list(active)
                cents = {nid: np.argwhere(nodes[nid].mask).mean(axis=0) for nid in acts}
                best = None
                best_d = np.inf
                for i in range(len(acts)):
                    for j in range(i + 1, len(acts)):
                        d = float(np.linalg.norm(cents[acts[i]] - cents[acts[j]]))
                        if d < best_d:
                            best_d, best = d, (acts[i], acts[j])
                u, v = best
                v_uv = self._value(
                    x, b, nodes[u].mask | nodes[v].mask, x0_val, target
                )
            else:
                if self.merge_mode == "color":
                    # Legacy: pure blind linkage, no model-aware selection.
                    u, v = min(adj, key=blind_dist)
                    v_uv = self._value(
                        x, b, nodes[u].mask | nodes[v].mask, x0_val, target
                    )
                else:
                    # Objective-driven: blind shortlist -> model-aware argmin.
                    shortlist = sorted(adj, key=blind_dist)[: max(1, self.shortlist_k)]
                    best = None
                    best_score = np.inf
                    best_vuv = 0.0
                    for (p, q) in shortlist:
                        merged_mask = nodes[p].mask | nodes[q].mask
                        v_pq = self._value(x, b, merged_mask, x0_val, target)
                        resid = v_pq - nodes[p].v - nodes[q].v
                        score = weight(int(merged_mask.sum())) * abs(resid)
                        if score < best_score:
                            best_score = score
                            best = (p, q)
                            best_vuv = v_pq
                    u, v = best
                    v_uv = best_vuv  # reuse the chosen trial -> no re-evaluation

            merged_mask = nodes[u].mask | nodes[v].mask
            merged = _Node(
                id=next_id,
                mask=merged_mask,
                children=[nodes[u], nodes[v]],
                v=v_uv,  # cached: parent value is the chosen trial value
            )
            nodes[next_id] = merged
            desc[next_id] = self._region_descriptor(desc_field, merged_mask)

            active.discard(u)
            active.discard(v)
            active.add(next_id)

            # Rebuild adjacency referencing the merged node.
            new_adj = set()
            for (p, q) in adj:
                if p in (u, v):
                    p = next_id
                if q in (u, v):
                    q = next_id
                if p != q and p in active and q in active:
                    new_adj.add((min(p, q), max(p, q)))
            adj = new_adj
            next_id += 1

        root_id = next(iter(active))
        return nodes[root_id]

    # ------------------------------------------------------------------ #
    # synergy fill (values are already cached during the build)
    # ------------------------------------------------------------------ #
    def _fill_deltas(self, root: _Node) -> None:
        """Set Delta(R) = v(R) - sum_j v(c_j) for every internal node.

        Values v(.) are cached on every node by _build_tree, so this is pure
        arithmetic -- no further forward passes.
        """
        stack = [root]
        while stack:
            n = stack.pop()
            if not n.is_leaf:
                n.delta = n.v - sum(c.v for c in n.children)
                stack.extend(n.children)

    # ------------------------------------------------------------------ #
    # explain
    # ------------------------------------------------------------------ #
    def explain(self, x: torch.Tensor) -> AttributionResult:
        x = x.to(self.device)
        target = self._resolve_target(x)
        self._build_queries = 0

        # On-manifold baseline / completion field: the strong-blur self-reference.
        b = blur_reference(x, self.sigma).to(self.device)

        img01 = denormalize(x)[0].permute(1, 2, 0).cpu().numpy()  # (H,W,3) in [0,1]
        H, W = img01.shape[:2]

        # f(x), f(x0) = f(b).
        f_x = self._target_prob(x, target)
        f_b = self._target_prob(b, target)  # x0 = full-blur reveal of nothing
        x0_val = f_b

        # Leaf segmentation.
        labels = self._leaf_labels(img01)

        # Blind proposal descriptor field: backbone features if available, else
        # raw color. Model-BLIND (no target class, no v) by construction.
        desc_field = None
        if self.merge_mode != "color":
            desc_field = self._feature_map(x, H, W)
        feature_used = desc_field is not None
        if desc_field is None:
            desc_field = img01  # (H,W,3) color fallback

        # Build the objective-driven tree (values cached during build).
        root = self._build_tree(img01, labels, x, b, x0_val, target, desc_field)

        # Fill Delta from cached values (no extra forward passes).
        self._fill_deltas(root)

        # --- completeness-style identity check (telescoping) ---------------- #
        leaves: list[_Node] = []
        internals: list[_Node] = []

        def split(n: _Node):
            (leaves if n.is_leaf else internals).append(n)
            for c in n.children:
                split(c)

        split(root)
        leaf_masks = {leaf.id: leaf.mask for leaf in leaves}

        sum_leaf_v = float(sum(l.v for l in leaves))
        sum_delta = float(sum(r.delta for r in internals))
        identity_lhs = float(root.v)                 # v(root)
        identity_rhs = sum_leaf_v + sum_delta        # sum leaf v + sum Delta
        identity_residual = identity_lhs - identity_rhs  # ~0 up to float error

        # --- merge-objective diagnostics ----------------------------------- #
        # J(T) value and the non-additivity index, for reporting / tree-sensitivity.
        def weight(area: int) -> float:
            return 1.0 / (float(area) + self.weight_eps)

        J_value = float(sum(weight(r.area) * abs(r.delta) for r in internals))
        nai_denom = sum_leaf_v_abs = float(sum(abs(l.v) for l in leaves)) + float(
            sum(abs(r.delta) for r in internals)
        )
        nai = (
            float(sum(abs(r.delta) for r in internals)) / nai_denom
            if nai_denom > 0
            else 0.0
        )

        # --- per-pixel attribution: leaf-additive density ------------------- #
        attr = np.zeros((H, W), dtype=np.float64)
        for leaf in leaves:
            area = max(leaf.area, 1)
            attr[leaf.mask] = leaf.v / area

        # Root reveal value via Phi, for parity with summary diagnostics.
        f_phi = self._target_prob(self._phi(x, b, root.mask), target)

        # --- serialize tree ------------------------------------------------- #
        def serialize(n: _Node) -> dict:
            return {
                "id": n.id,
                "area": n.area,
                "v": float(n.v),
                "delta": float(n.delta),
                "is_leaf": n.is_leaf,
                "child_ids": [c.id for c in n.children],
            }

        all_serialized = []

        def walk(n: _Node):
            all_serialized.append(serialize(n))
            for c in n.children:
                walk(c)

        walk(root)

        # Forward-pass accounting. The clean lower bound is 2n-1 (one value per
        # node); the objective-driven build spends extra on rejected shortlist
        # trials. _build_queries counts every v(.) evaluated during the build.
        n_leaves = len(leaves)
        budget_2n_minus_1 = 2 * n_leaves - 1

        return AttributionResult(
            attribution=attr,
            method=self.name,
            target_class=target,
            target_class_name=self._class_name(target),
            f_x=f_x,
            f_b=f_b,
            f_phi=f_phi,
            extras={
                "sigma": self.sigma,
                "n_segments": self.n_segments,
                "n_leaves": n_leaves,
                "n_internal": len(internals),
                # --- value-query accounting --- #
                "n_value_queries": self._build_queries,  # incl. rejected trials
                "budget_2n_minus_1": budget_2n_minus_1,   # clean lower bound
                # --- merge-objective settings + diagnostics --- #
                "merge_mode": self.merge_mode,
                "shortlist_k": self.shortlist_k,
                "feature_layer": self.feature_layer if feature_used else None,
                "blind_signal": "feature" if feature_used else "color",
                "weight_eps": self.weight_eps,
                "J_objective": J_value,                  # sum_R w(R)|Delta(R)|
                "NAI": nai,                              # non-additivity index
                # --- tree + identity --- #
                "root_v": float(root.v),
                "sum_leaf_v": sum_leaf_v,
                "sum_delta": sum_delta,
                "identity_lhs": identity_lhs,
                "identity_rhs": identity_rhs,
                "identity_residual": identity_residual,  # telescoping check ~0
                "tree": all_serialized,                  # full v/Delta per node
                "leaf_masks": leaf_masks,
                "reference": "blur_completion",
            },
        )