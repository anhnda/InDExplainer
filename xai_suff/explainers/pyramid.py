"""PyramidExplainer -- model-aware hierarchical EVIDENCE LOCALIZATION.

This is a re-derivation of the method. The previous version built the region
tree as the greedy MINIMIZER of non-additivity J(T) = sum_R w(R)|Delta(R)|,
which structurally defers every value-gaining merge upward: the object region
never forms as a clean node, value only completes at the root, and Delta(root)
absorbs essentially the whole response ("root takes all"). That objective is
wrong for the question we actually care about.

New question
------------
    Which SMALL, compact superregion does the model actually rely on, and can
    we grow it as a single node whose value already equals (almost) the whole
    response -- so that revealing it occupies the main focus of the object,
    the remainder from that superregion down to its leaves telescopes cleanly,
    and the final merge with the (uninformative) background is a near-zero
    remainder rather than the dominant term?

Construction (value-gain agglomeration)
---------------------------------------
On-manifold value of a region R (blur-completion Phi):

    Phi_R(x) = reveal R sharp, complement = blur_sigma(x)
    v(R)     = z_y(Phi_R(x)) - z_y(b)              (logits; b = blurred self-ref)

At each step we merge the adjacent pair whose union gains the most target value
PER PIXEL ADDED:

    gain(u,v) = [ v(u|v) - max(v(u), v(v)) ] / area(u|v)
    (u*,v*)   = argmax_{(u,v) in adj} gain(u,v)                          (MAXIMIZE)

Rationale. Regions that JOINTLY cross the recognition threshold raise v sharply
and merge early, snapping the object together into a compact high-v node R*.
Background regions (sky, grass) raise v by ~0 per pixel, so they stay low-v and
merge last. The object superregion therefore appears as a real INTERNAL node
with v(R*) ~= v(root); the final root merge (object + background) carries a
small Delta. This is the opposite of the old minimize-|Delta| direction, and it
is what "the superregion takes most of the score, not the root" demands.

Blind proposal, model-aware selection. A pure greedy max would trial v(u|v) for
every adjacent pair at every level (O(nE) passes). We shortlist the k most
backbone-feature-similar adjacent pairs (model-BLIND: no target, no v), then
score only those k by the value-gain criterion. The chosen union value is cached
and reused as the node value, so only the <= k-1 rejected trials cost extra
passes => O(n*k).

Focal node. After the build we report the FOCAL node R* = the minimal-area node
with v(R*) >= tau * v(root). Its subtree is the model's evidence; everything
above it is remainder. (Reported in extras; also drives the per-pixel map.)

Conservation (preserved, holds for ANY tree)
--------------------------------------------
    v(root) = sum_{leaves} v(leaf) + sum_{internal} Delta(R),   Delta = v(R) - sum v(child)

The telescoping identity is independent of the merge rule, so the value-gain
construction changes only WHICH coalitions are sampled, never the correctness of
the accounting. The reconstruction residual is reported as a self-check.

Per-pixel attribution
----------------------
The (H,W) map is the FOCAL-subtree leaf density: leaves under R* take
v(leaf)/area(leaf); leaves outside R* are zeroed (they are remainder, not
evidence). This makes the map point at the compact object region. The raw
leaf-additive density over all leaves is also provided in extras for parity.

Reference dependence (Phi, tree, tau) is logged in extras, not hidden.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F

try:  # scikit-image is used for the leaf segmentation
    from skimage.segmentation import slic
    _HAS_SKIMAGE = True
except Exception:  # pragma: no cover
    _HAS_SKIMAGE = False

from .base import AttributionResult, Explainer, blur_reference, denormalize


@dataclass
class _Node:
    """One region in the agglomerative tree."""
    id: int
    mask: np.ndarray            # (H,W) bool
    children: list              # list[_Node]; empty for leaves
    v: float = 0.0              # holistic on-manifold value z_y(Phi_R) - z_y(b)
    delta: float = 0.0          # synergy v(R) - sum_j v(child_j); 0 for leaves

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
        sigma: float = 11.0,            # blur strength for Phi complement
        n_segments: int = 144,          # target number of leaf superpixels (SLIC)
        compactness: float = 2.0,       # SLIC compactness
        # --- value-gain merge controls ---------------------------------- #
        shortlist_k: int = 6,           # blind-proposal shortlist size per step
        feature_layer: str = "layer3",  # backbone layer for blind feature dist
        merge_mode: str = "focus",      # "focus" (value-gain) | "color" (legacy blind)
        use_logits: bool = True,        # value in logit space (matches the paper)
        focal_tau: float = 0.8,         # focal node threshold: v(R*) >= tau*v(root)
        gain_area_pow: float = 1.0,     # area exponent in the per-pixel denominator
        **kw,
    ):
        super().__init__(*args, **kw)
        self.sigma = sigma
        self.n_segments = n_segments
        self.compactness = compactness
        self.shortlist_k = shortlist_k
        self.feature_layer = feature_layer
        self.merge_mode = merge_mode
        self.use_logits = use_logits
        self.focal_tau = focal_tau
        self.gain_area_pow = gain_area_pow
        self._build_queries = 0

    # ------------------------------------------------------------------ #
    # Phi: blur-completion manifold-projection operator
    # ------------------------------------------------------------------ #
    def _phi(self, x: torch.Tensor, b: torch.Tensor, mask_bool: np.ndarray) -> torch.Tensor:
        m = torch.as_tensor(mask_bool, dtype=x.dtype, device=x.device)
        m = m.view(1, 1, *mask_bool.shape)
        return m * x + (1.0 - m) * b

    def _target_response(self, comp: torch.Tensor, target: int) -> float:
        """Target-class response. Logit (default) or probability."""
        with torch.no_grad():
            out = self.model(comp)
            if self.use_logits:
                return float(out[:, target].item())
            return float(F.softmax(out, dim=1)[:, target].item())

    def _value(self, x, b, mask_bool, x0_val: float, target: int) -> float:
        """v(R) = z_y(Phi_R(x)) - z_y(b), counting one forward pass."""
        self._build_queries += 1
        return self._target_response(self._phi(x, b, mask_bool), target) - x0_val

    # ------------------------------------------------------------------ #
    # leaf segmentation
    # ------------------------------------------------------------------ #
    def _leaf_labels(self, img01: np.ndarray) -> np.ndarray:
        if _HAS_SKIMAGE:
            labels = slic(
                img01,
                n_segments=self.n_segments,
                compactness=self.compactness,
                start_label=0,
                channel_axis=2,
            )
            return labels.astype(np.int64)
        H, W = img01.shape[:2]
        side = max(1, int(round(np.sqrt(self.n_segments))))
        ys = np.linspace(0, side, H, endpoint=False).astype(np.int64)
        xs = np.linspace(0, side, W, endpoint=False).astype(np.int64)
        return (ys[:, None] * side + xs[None, :]).astype(np.int64)

    # ------------------------------------------------------------------ #
    # blind proposal signal: backbone intermediate features (no target, no v)
    # ------------------------------------------------------------------ #
    def _feature_map(self, x: torch.Tensor, H: int, W: int) -> Optional[np.ndarray]:
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
        act = F.interpolate(act, size=(H, W), mode="bilinear", align_corners=False)
        return act[0].permute(1, 2, 0).cpu().numpy().astype(np.float64)

    @staticmethod
    def _region_descriptor(desc_field: np.ndarray, mask: np.ndarray) -> np.ndarray:
        return desc_field[mask].mean(axis=0)

    # ------------------------------------------------------------------ #
    # value-gain agglomerative tree
    # ------------------------------------------------------------------ #
    def _build_tree(
        self,
        labels: np.ndarray,
        x: torch.Tensor,
        b: torch.Tensor,
        x0_val: float,
        target: int,
        desc_field: np.ndarray,
    ) -> _Node:
        H, W = labels.shape
        uniq = np.unique(labels)

        next_id = 0
        nodes: dict[int, _Node] = {}
        for lab in uniq:
            mask = labels == lab
            node = _Node(id=next_id, mask=mask, children=[])
            node.v = self._value(x, b, mask, x0_val, target)  # leaf value, cached
            nodes[next_id] = node
            next_id += 1

        active = set(nodes.keys())
        desc = {nid: self._region_descriptor(desc_field, nodes[nid].mask) for nid in active}

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

        def gain(v_uv: float, v_u: float, v_v: float, area: int) -> float:
            # value gained per pixel added: reward crossing the threshold,
            # reward compactness (small area). Higher is better.
            denom = float(area) ** self.gain_area_pow
            return (v_uv - max(v_u, v_v)) / max(denom, 1.0)

        # Greedy MAXIMIZATION of value-gain. Each step: blind shortlist ->
        # model-aware argmax of gain. The object region assembles first.
        while len(active) > 1:
            if not adj:
                # Disconnected remainder (rare): merge spatially-closest pair.
                acts = list(active)
                cents = {nid: np.argwhere(nodes[nid].mask).mean(axis=0) for nid in acts}
                best, best_d = None, np.inf
                for i in range(len(acts)):
                    for j in range(i + 1, len(acts)):
                        d = float(np.linalg.norm(cents[acts[i]] - cents[acts[j]]))
                        if d < best_d:
                            best_d, best = d, (acts[i], acts[j])
                u, v = best
                v_uv = self._value(x, b, nodes[u].mask | nodes[v].mask, x0_val, target)
            elif self.merge_mode == "color":
                # Legacy pure-blind linkage (no model-aware selection).
                u, v = min(adj, key=blind_dist)
                v_uv = self._value(x, b, nodes[u].mask | nodes[v].mask, x0_val, target)
            else:
                # focus mode: blind shortlist -> model-aware MAX value-gain.
                shortlist = sorted(adj, key=blind_dist)[: max(1, self.shortlist_k)]
                best, best_score, best_vuv = None, -np.inf, 0.0
                for (p, q) in shortlist:
                    merged_mask = nodes[p].mask | nodes[q].mask
                    v_pq = self._value(x, b, merged_mask, x0_val, target)
                    score = gain(v_pq, nodes[p].v, nodes[q].v, int(merged_mask.sum()))
                    if score > best_score:
                        best_score, best, best_vuv = score, (p, q), v_pq
                u, v = best
                v_uv = best_vuv  # reuse chosen trial -> no re-evaluation

            merged_mask = nodes[u].mask | nodes[v].mask
            merged = _Node(id=next_id, mask=merged_mask,
                           children=[nodes[u], nodes[v]], v=v_uv)
            nodes[next_id] = merged
            desc[next_id] = self._region_descriptor(desc_field, merged_mask)

            active.discard(u)
            active.discard(v)
            active.add(next_id)

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

        return nodes[next(iter(active))]

    # ------------------------------------------------------------------ #
    # synergy fill (values cached during build)
    # ------------------------------------------------------------------ #
    def _fill_deltas(self, root: _Node) -> None:
        stack = [root]
        while stack:
            n = stack.pop()
            if not n.is_leaf:
                n.delta = n.v - sum(c.v for c in n.children)
                stack.extend(n.children)

    # ------------------------------------------------------------------ #
    # focal node: minimal-area node with v(R*) >= tau * v(root)
    # ------------------------------------------------------------------ #
    @staticmethod
    def _find_focal(root: _Node, tau: float) -> _Node:
        target_v = tau * root.v
        best = root
        best_area = root.area
        stack = [root]
        while stack:
            n = stack.pop()
            if n.v >= target_v and n.area <= best_area:
                best, best_area = n, n.area
            stack.extend(n.children)
        return best

    # ------------------------------------------------------------------ #
    # explain
    # ------------------------------------------------------------------ #
    def explain(self, x: torch.Tensor) -> AttributionResult:
        x = x.to(self.device)
        target = self._resolve_target(x)
        self._build_queries = 0

        b = blur_reference(x, self.sigma).to(self.device)
        img01 = denormalize(x)[0].permute(1, 2, 0).cpu().numpy()
        H, W = img01.shape[:2]

        f_x = self._target_response(x, target)
        f_b = self._target_response(b, target)
        x0_val = f_b

        labels = self._leaf_labels(img01)

        desc_field = None
        if self.merge_mode != "color":
            desc_field = self._feature_map(x, H, W)
        feature_used = desc_field is not None
        if desc_field is None:
            desc_field = img01

        root = self._build_tree(labels, x, b, x0_val, target, desc_field)
        self._fill_deltas(root)

        # ---- collect nodes ------------------------------------------------ #
        leaves: list[_Node] = []
        internals: list[_Node] = []

        def split(n: _Node):
            (leaves if n.is_leaf else internals).append(n)
            for c in n.children:
                split(c)

        split(root)
        leaf_masks = {leaf.id: leaf.mask for leaf in leaves}

        # ---- focal node (the compact evidence superregion) ---------------- #
        focal = self._find_focal(root, self.focal_tau)
        focal_leaves = []

        def collect_focal_leaves(n: _Node):
            if n.is_leaf:
                focal_leaves.append(n)
            for c in n.children:
                collect_focal_leaves(c)

        collect_focal_leaves(focal)

        # ---- conservation self-check (telescoping) ------------------------ #
        sum_leaf_v = float(sum(l.v for l in leaves))
        sum_delta = float(sum(r.delta for r in internals))
        identity_lhs = float(root.v)
        identity_rhs = sum_leaf_v + sum_delta
        identity_residual = identity_lhs - identity_rhs

        # ---- diagnostics: is the score concentrated, or root-takes-all? --- #
        root_children = root.children
        root_child_v = [float(c.v) for c in root_children]
        root_v = float(root.v) if abs(root.v) > 1e-12 else 1e-12
        focal_fraction = float(focal.v) / root_v          # want ~>= tau
        focal_area_frac = focal.area / float(H * W)        # want small
        root_delta_frac = float(root.delta) / root_v       # want ~0
        # NAI for parity with the old report.
        nai_denom = float(sum(abs(l.v) for l in leaves)) + float(
            sum(abs(r.delta) for r in internals))
        nai = (float(sum(abs(r.delta) for r in internals)) / nai_denom
               if nai_denom > 0 else 0.0)

        # ---- per-pixel attribution: FOCAL-subtree leaf density ------------ #
        attr = np.zeros((H, W), dtype=np.float64)
        for leaf in focal_leaves:
            attr[leaf.mask] = leaf.v / max(leaf.area, 1)

        # full leaf-additive density (parity / comparison)
        attr_all = np.zeros((H, W), dtype=np.float64)
        for leaf in leaves:
            attr_all[leaf.mask] = leaf.v / max(leaf.area, 1)

        f_phi = self._target_response(self._phi(x, b, root.mask), target)

        # ---- serialize tree ----------------------------------------------- #
        def serialize(n: _Node) -> dict:
            return {
                "id": n.id, "area": n.area,
                "v": float(n.v), "delta": float(n.delta),
                "is_leaf": n.is_leaf,
                "child_ids": [c.id for c in n.children],
                "is_focal": n.id == focal.id,
            }

        all_serialized = []

        def walk(n: _Node):
            all_serialized.append(serialize(n))
            for c in n.children:
                walk(c)

        walk(root)

        n_leaves = len(leaves)

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
                "value_space": "logit" if self.use_logits else "prob",
                # --- value-query accounting --- #
                "n_value_queries": self._build_queries,
                "budget_2n_minus_1": 2 * n_leaves - 1,
                # --- merge settings --- #
                "merge_mode": self.merge_mode,
                "shortlist_k": self.shortlist_k,
                "feature_layer": self.feature_layer if feature_used else None,
                "blind_signal": "feature" if feature_used else "color",
                "gain_area_pow": self.gain_area_pow,
                # --- focal node: the compact evidence superregion --- #
                "focal_tau": self.focal_tau,
                "focal_id": focal.id,
                "focal_v": float(focal.v),
                "focal_area": focal.area,
                "focal_area_frac": focal_area_frac,     # want SMALL
                "focal_fraction": focal_fraction,       # v(R*)/v(root), want >= tau
                "focal_n_leaves": len(focal_leaves),
                # --- root-takes-all diagnostics --- #
                "root_v": root_v,
                "root_child_v": root_child_v,           # both should be substantial
                "root_delta": float(root.delta),
                "root_delta_frac": root_delta_frac,     # want ~0, NOT >0.5
                # --- conservation --- #
                "sum_leaf_v": sum_leaf_v,
                "sum_delta": sum_delta,
                "identity_lhs": identity_lhs,
                "identity_rhs": identity_rhs,
                "identity_residual": identity_residual,
                "NAI": nai,
                # --- maps + tree --- #
                "attr_all_leaves": attr_all,
                "tree": all_serialized,
                "leaf_masks": leaf_masks,
                "reference": "blur_completion",
            },
        )