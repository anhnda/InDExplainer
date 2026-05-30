"""PyramidExplainer -- hierarchical, interaction-aware, on-manifold attribution.

Implements the method note (PyramidExplainer.md) against the package contract:

    Explainer(model, target_class=None, device=...)
    explainer.explain(x) -> AttributionResult

Core construction
-----------------
1. Segment the image into superpixels (leaves).
2. Build an agglomerative region tree T by repeatedly merging the most similar
   adjacent regions (leaves -> ... -> root = whole image).
3. Holistic on-manifold value of any region R (blur-completion Phi):

       Phi_R(x) = reveal R sharp, replace complement R^c with blur_sigma(x)
       v(R)     = f(Phi_R(x)) - f(x0)            with x0 = full blur reference b

4. Node synergy (whole minus parts) for internal node R with children c_1..c_m:

       Delta(R) = v(R) - sum_j v(c_j)

   Delta>0 cooperation, Delta<0 redundancy.

5. Completeness-style identity (telescoping):

       v(root) = sum_{leaves} v(leaf) + sum_{internal} Delta(R)

Per-pixel attribution
----------------------
The returned (H,W) map is the leaf-additive part: each pixel takes the value of
its leaf superpixel, v(leaf) / area(leaf) (a per-pixel density of the leaf's
holistic value). Interaction is reported separately in `extras` (the full tree
with each node's v and Delta), since "synergy at scale s" is tree-relative and
not a per-pixel quantity (see method note section 4, tree-dependence caveat).

Reference dependence (Phi, tree) is stated, not hidden: the chosen sigma and
segmentation parameters are logged in `extras`.
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
        n_segments: int = 64,       # target number of leaf superpixels (SLIC)
        compactness: float = 10.0,  # SLIC compactness
        max_nodes: Optional[int] = None,  # cap on internal-node value queries
        **kw,
    ):
        super().__init__(*args, **kw)
        self.sigma = sigma
        self.n_segments = n_segments
        self.compactness = compactness
        self.max_nodes = max_nodes

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
    # agglomerative region tree (merge most-similar adjacent regions)
    # ------------------------------------------------------------------ #
    def _build_tree(self, img01: np.ndarray, labels: np.ndarray) -> _Node:
        H, W = labels.shape
        uniq = np.unique(labels)

        # Leaf nodes.
        next_id = 0
        nodes: dict[int, _Node] = {}
        for lab in uniq:
            mask = labels == lab
            nodes[next_id] = _Node(id=next_id, mask=mask, children=[])
            next_id += 1

        active = set(nodes.keys())

        # Region mean color and adjacency.
        def mean_color(mask: np.ndarray) -> np.ndarray:
            return img01[mask].mean(axis=0)

        colors = {nid: mean_color(nodes[nid].mask) for nid in active}

        # Adjacency among current active regions.
        def adjacency(masks: dict[int, np.ndarray]) -> set:
            # Two regions are adjacent if any 4-neighbour pixels differ in owner.
            owner = -np.ones((H, W), dtype=np.int64)
            for nid, m in masks.items():
                owner[m] = nid
            pairs = set()
            # horizontal & vertical neighbour comparisons
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

        # Greedy agglomeration: at each step merge the adjacent pair with the
        # smallest mean-color distance, until a single root remains.
        while len(active) > 1:
            if not adj:
                # Disconnected remainder: merge arbitrary two actives.
                it = iter(active)
                u = next(it)
                v = next(it)
            else:
                u, v = min(
                    adj,
                    key=lambda p: float(np.linalg.norm(colors[p[0]] - colors[p[1]])),
                )

            merged_mask = nodes[u].mask | nodes[v].mask
            merged = _Node(
                id=next_id,
                mask=merged_mask,
                children=[nodes[u], nodes[v]],
            )
            nodes[next_id] = merged
            colors[next_id] = mean_color(merged_mask)

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
    # value + synergy computation over the tree
    # ------------------------------------------------------------------ #
    def _compute_values(self, root: _Node, x, b, x0_val: float, target: int):
        """Fill v(R) for every node, then Delta(R) for internal nodes.

        x0_val = f(x0) = f(b): the baseline subtracted in v(R) = f(Phi_R) - f(x0).
        Returns (n_value_queries,).
        """
        # Gather nodes (optionally cap internal queries by area-priority).
        all_nodes: list[_Node] = []

        def collect(n: _Node):
            all_nodes.append(n)
            for c in n.children:
                collect(c)

        collect(root)

        # Value query for each node: v(R) = f(Phi_R(x)) - f(x0).
        n_queries = 0
        for node in all_nodes:
            f_phi = self._target_prob(self._phi(x, b, node.mask), target)
            node.v = f_phi - x0_val
            n_queries += 1

        # Synergy (whole minus parts), bottom-up implicit (children already have v).
        for node in all_nodes:
            if not node.is_leaf:
                node.delta = node.v - sum(c.v for c in node.children)

        return n_queries

    # ------------------------------------------------------------------ #
    # explain
    # ------------------------------------------------------------------ #
    def explain(self, x: torch.Tensor) -> AttributionResult:
        x = x.to(self.device)
        target = self._resolve_target(x)

        # On-manifold baseline / completion field: the strong-blur self-reference.
        b = blur_reference(x, self.sigma).to(self.device)

        img01 = denormalize(x)[0].permute(1, 2, 0).cpu().numpy()  # (H,W,3) in [0,1]
        H, W = img01.shape[:2]

        # f(x), f(x0) = f(b).
        f_x = self._target_prob(x, target)
        f_b = self._target_prob(b, target)  # this is x0 = full-blur reveal of nothing
        x0_val = f_b

        # Build leaf segmentation and agglomerative tree.
        labels = self._leaf_labels(img01)
        root = self._build_tree(img01, labels)

        # Fill v(R) and Delta(R) across the tree.
        n_queries = self._compute_values(root, x, b, x0_val, target)

        # --- completeness-style identity check (telescoping) ---------------- #
        leaves: list[_Node] = []
        internals: list[_Node] = []

        def split(n: _Node):
            (leaves if n.is_leaf else internals).append(n)
            for c in n.children:
                split(c)

        split(root)

        sum_leaf_v = float(sum(l.v for l in leaves))
        sum_delta = float(sum(r.delta for r in internals))
        identity_lhs = float(root.v)                 # v(root)
        identity_rhs = sum_leaf_v + sum_delta        # sum leaf v + sum Delta
        identity_residual = identity_lhs - identity_rhs  # ~0 up to float error

        # --- per-pixel attribution: leaf-additive density ------------------- #
        # Each pixel takes its leaf's holistic value spread over the leaf area,
        # so summing the map recovers sum_{leaves} v(leaf) (the additive part).
        attr = np.zeros((H, W), dtype=np.float64)
        for leaf in leaves:
            area = max(leaf.area, 1)
            attr[leaf.mask] = leaf.v / area

        # f_phi reported as the root reveal value (whole image sharp) for parity
        # with the summary panel diagnostics; equals f(x) when the root mask is
        # the full frame, but is computed via Phi for consistency.
        f_phi = self._target_prob(self._phi(x, b, root.mask), target)

        # Serialize the tree (id, area, v, delta, child ids) for downstream
        # interaction analysis -- synergy is tree-relative, not per-pixel.
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
                "n_leaves": len(leaves),
                "n_internal": len(internals),
                "n_value_queries": n_queries,
                "root_v": float(root.v),
                "sum_leaf_v": sum_leaf_v,
                "sum_delta": sum_delta,
                "identity_lhs": identity_lhs,
                "identity_rhs": identity_rhs,
                "identity_residual": identity_residual,  # telescoping check ~0
                "tree": all_serialized,                  # full v/Delta per node
                "reference": "blur_completion",
            },
        )