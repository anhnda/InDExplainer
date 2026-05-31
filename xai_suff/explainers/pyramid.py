"""PyramidExplainer -- model-aware agglomerative hierarchical residuals.

The region tree is built bottom-up by *merging the adjacent region pair the
model treats as most non-additive*, rather than by a model-blind color
heuristic or a fixed leaf-by-leaf growth chain. Construction and explanation
use the same quantity: we merge on the merge residual Delta and we report it.

Merge criterion (set by `merge_criterion`)
------------------------------------------
At each step, over adjacent pairs (Ri, Rj) of the current active regions:

    Delta(Ri, Rj) = v(Ri u Rj) - v(Ri) - v(Rj)

    - "max_coop": merge argmax Delta        (surface cooperation first)
    - "max_both": merge argmax |Delta|      (surface cooperation AND
                                             redundancy/suppression first)

v(Ri), v(Rj) are cached; only v(Ri u Rj) costs a fresh forward. The winning
pair fuses into a binary internal node; its value is the union value just
computed and its stored residual is the winning Delta. After n-1 merges the
single remaining region is the root. Children at every node disjointly
partition the parent, so Theorem 1 (telescoping) holds unchanged:

    v(root) = sum_{leaves} v(leaf) + sum_{internal} Delta(R)

Cost
----
Heap-cached agglomeration: each merge only introduces fresh pairs between the
new node and its neighbours (planar adjacency, avg degree small), so the total
*fresh* forward count is ~ n (leaf values) + sum of new-neighbour counts
~= O(n) in practice, not O(n^2). `frontier_k` caps pairs scored per step.

Caveat (mandatory experiment, not optional)
--------------------------------------------
The tree is now model-aware and can flatter the model: a tree built to surface
cooperation will show cooperation. Random-tree and color-tree controls are
REQUIRED to demonstrate that the residual structure is a property of the model,
not of a flattering hierarchy. Reference (Phi, blur sigma) and query counts are
logged, not hidden.

Per-pixel attribution remains the leaf-additive density v(leaf)/area(leaf);
Delta and the sufficiency node are reported in `extras` (tree-relative).
"""
from __future__ import annotations

import heapq
import itertools
from dataclasses import dataclass
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
    """One region in the region tree."""
    id: int
    mask: np.ndarray            # (H,W) bool, pixels belonging to this region
    children: list              # list[_Node]; empty for leaves
    v: float = 0.0              # holistic on-manifold value f(Phi_R(x)) - f(x0)
    delta: float = 0.0          # merge residual v(R) - sum_j v(child_j); 0 for leaves

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
        sigma: float = 11.0,              # blur strength for Phi complement
        n_segments: int = 144,            # target number of leaf superpixels (SLIC)
        compactness: float = 2,           # SLIC compactness
        merge_criterion: str = "max_both",  # "max_coop" (Delta) or "max_both" (|Delta|)
        frontier_k: Optional[int] = None,   # cap candidate pairs scored per step
        suff_eps: float = 0.05,           # sufficiency: v(S) >= (1-eps) v(root)
        **kw,
    ):
        super().__init__(*args, **kw)
        if merge_criterion not in ("max_coop", "max_both"):
            raise ValueError(
                f"merge_criterion must be 'max_coop' or 'max_both', got {merge_criterion!r}"
            )
        self.sigma = sigma
        self.n_segments = n_segments
        self.compactness = compactness
        self.merge_criterion = merge_criterion
        self.frontier_k = frontier_k
        self.suff_eps = suff_eps

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

    # value of a region given by a boolean mask: v(R) = f(Phi_R) - x0_val
    def _value_of_mask(self, mask_bool, x, b, x0_val: float, target: int) -> float:
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
    # leaf-level adjacency (computed once on the fixed leaf partition)
    # ------------------------------------------------------------------ #
    def _leaf_adjacency(self, labels: np.ndarray) -> dict[int, set]:
        """Return {leaf_label: set(neighbour leaf_labels)} via 4-connectivity."""
        adj: dict[int, set] = {int(l): set() for l in np.unique(labels)}
        a, b_ = labels[:, :-1], labels[:, 1:]
        diff = a != b_
        for u, v in zip(a[diff].ravel(), b_[diff].ravel()):
            adj[int(u)].add(int(v))
            adj[int(v)].add(int(u))
        a, b_ = labels[:-1, :], labels[1:, :]
        diff = a != b_
        for u, v in zip(a[diff].ravel(), b_[diff].ravel()):
            adj[int(u)].add(int(v))
            adj[int(v)].add(int(u))
        return adj

    # ------------------------------------------------------------------ #
    # priority key for the merge criterion
    # ------------------------------------------------------------------ #
    def _merge_key(self, delta: float) -> float:
        """Higher key == merge sooner. heapq is a min-heap, so we negate."""
        if self.merge_criterion == "max_coop":
            return delta          # most positive cooperation first
        else:  # max_both
            return abs(delta)     # strongest |non-additivity| (coop OR suppression)

    # ------------------------------------------------------------------ #
    # region tree by model-aware agglomerative merging
    # ------------------------------------------------------------------ #
    def _build_tree(self, img01, labels, x, b, x0_val: float, target: int):
        """Bottom-up agglomeration: repeatedly fuse the adjacent active-region
        pair with the largest merge key (Delta or |Delta|).

        Returns (root, info) with construction diagnostics: merge order, the
        residual realised at each merge, and fresh forward-query count.

        Heap-caching: a candidate pair's Delta is valid until one of its two
        regions is consumed by a merge. We push (-key, tiebreak, ida, idb,
        delta, v_union) entries and lazily skip stale ones (regions no longer
        active). After each merge only the new node's pairs are scored.
        """
        uniq = [int(l) for l in np.unique(labels)]
        n = len(uniq)

        leaf_mask = {l: (labels == l) for l in uniq}
        leaf_color = {l: img01[leaf_mask[l]].mean(axis=0) for l in uniq}
        leaf_adj = self._leaf_adjacency(labels)

        n_fresh_q = 0  # fresh model forwards spent in construction

        # ---- node registry -------------------------------------------- #
        node: dict[int, _Node] = {}
        next_id = 0

        # Leaf nodes (ids 0..n-1). Leaf value v({l}) = one forward each.
        leaf_node: dict[int, _Node] = {}
        leaf_value: dict[int, float] = {}
        for l in uniq:
            leaf_value[l] = self._value_of_mask(leaf_mask[l], x, b, x0_val, target)
            n_fresh_q += 1
            nd = _Node(id=next_id, mask=leaf_mask[l], children=[])
            nd.v = leaf_value[l]
            node[next_id] = nd
            leaf_node[l] = nd
            next_id += 1

        # ---- active-region bookkeeping (keyed by node id) ------------- #
        active: set = set(leaf_node[l].id for l in uniq)         # live region ids
        nadj: dict[int, set] = {                                  # region adjacency
            leaf_node[l].id: set(leaf_node[m].id for m in leaf_adj[l])
            for l in uniq
        }
        # leaf-color per active region id (for cheap background tie-handling only)
        rcolor: dict[int, np.ndarray] = {leaf_node[l].id: leaf_color[l] for l in uniq}

        # ---- candidate-pair heap -------------------------------------- #
        # entry: (negkey, tiebreak, ida, idb, delta, v_union)
        heap: list = []
        tiebreak = itertools.count()

        def score_pair(ida: int, idb: int):
            """Compute Delta for an active pair, push to heap. One forward."""
            nonlocal n_fresh_q
            na, nb = node[ida], node[idb]
            union_mask = na.mask | nb.mask
            v_union = self._value_of_mask(union_mask, x, b, x0_val, target)
            n_fresh_q += 1
            delta = v_union - na.v - nb.v
            key = self._merge_key(delta)
            heapq.heappush(heap, (-key, next(tiebreak), ida, idb, delta, v_union))

        # Optional frontier_k cap: rank a region's neighbours by a cheap
        # model-aware proxy (current region value) and only score the top-k
        # pairs for that region. Reduces forwards at the cost of approximating
        # the global argmax.
        def neighbours_to_score(rid: int) -> list:
            cand = [m for m in nadj[rid] if m in active]
            if self.frontier_k is not None and len(cand) > self.frontier_k:
                cand = sorted(cand, key=lambda m: node[m].v, reverse=True)
                cand = cand[: self.frontier_k]
            return cand

        # Seed the heap with every adjacent leaf pair (each pair once).
        seeded: set = set()
        for rid in list(active):
            for m in neighbours_to_score(rid):
                key2 = (min(rid, m), max(rid, m))
                if key2 in seeded:
                    continue
                seeded.add(key2)
                score_pair(key2[0], key2[1])

        merge_order = []   # list of (child_a_id, child_b_id, merged_id, delta)
        merge_deltas = []  # realised residual at each merge

        # ---- agglomeration loop --------------------------------------- #
        while len(active) > 1:
            # Pop the best *still-valid* candidate.
            best = None
            while heap:
                negkey, _, ida, idb, delta, v_union = heapq.heappop(heap)
                if ida in active and idb in active:
                    best = (ida, idb, delta, v_union)
                    break
                # else: stale (a region was consumed) -> discard

            if best is None:
                # Heap exhausted but >1 region remains: disconnected components
                # (rare). Join the two cheapest-to-evaluate leftover regions by
                # color proximity to finish the tree without spurious queries.
                left = list(active)
                # nearest color pair among leftovers
                pa, pb, bestd = left[0], left[1], np.inf
                for i in range(len(left)):
                    for j in range(i + 1, len(left)):
                        d = float(np.sum((rcolor[left[i]] - rcolor[left[j]]) ** 2))
                        if d < bestd:
                            bestd, pa, pb = d, left[i], left[j]
                ida, idb = pa, pb
                na, nb = node[ida], node[idb]
                union_mask = na.mask | nb.mask
                v_union = self._value_of_mask(union_mask, x, b, x0_val, target)
                n_fresh_q += 1
                delta = v_union - na.v - nb.v
                best = (ida, idb, delta, v_union)

            ida, idb, delta, v_union = best
            na, nb = node[ida], node[idb]

            # Create the merged binary node; reuse v_union (no re-query).
            merged = _Node(
                id=next_id,
                mask=(na.mask | nb.mask),
                children=[na, nb],
            )
            merged.v = v_union
            merged.delta = delta
            node[next_id] = merged
            mid = next_id
            next_id += 1

            # Update active set and adjacency: merged inherits both neighbour
            # sets (minus the two consumed regions).
            active.discard(ida)
            active.discard(idb)
            new_neigh = (nadj[ida] | nadj[idb]) - {ida, idb}
            new_neigh = {m for m in new_neigh if m in active}
            nadj[mid] = new_neigh
            for m in new_neigh:
                nadj[m].discard(ida)
                nadj[m].discard(idb)
                nadj[m].add(mid)
            active.add(mid)
            # area-weighted mean color for the merged region (tie-handling only)
            wa, wb = na.area, nb.area
            rcolor[mid] = (rcolor[ida] * wa + rcolor[idb] * wb) / max(wa + wb, 1)

            merge_order.append((ida, idb, mid, float(delta)))
            merge_deltas.append(float(delta))

            # Score the new node's pairs (only these are fresh).
            for m in neighbours_to_score(mid):
                score_pair(mid, m)

        root = node[next(iter(active))]

        # ---- sufficiency node: smallest-area node with v >= (1-eps)v_root #
        # Walk all nodes; among those clearing the threshold pick min area.
        v_root_est = root.v
        thresh = (1.0 - self.suff_eps) * v_root_est if v_root_est > 0 else v_root_est
        suff_node = root
        suff_area = root.area
        for nd in node.values():
            if nd.v >= thresh and nd.area < suff_area:
                suff_node, suff_area = nd, nd.area

        info = {
            "merge_order": merge_order,                 # (a,b,merged,delta) per step
            "merge_deltas": merge_deltas,               # realised residual per merge
            "v_root_est": float(v_root_est),
            "suff_node_id": int(suff_node.id),
            "suff_v": float(suff_node.v),
            "n_construction_queries": int(n_fresh_q),
            "merge_criterion": self.merge_criterion,
        }
        return root, info

    # ------------------------------------------------------------------ #
    # value + merge-residual computation over the tree
    # ------------------------------------------------------------------ #
    def _compute_values(self, root: _Node, x, b, x0_val: float, target: int,
                        skip_if_present: bool = True):
        """Fill v(R) for every node, then Delta(R) for internal nodes.

        Nodes built by _build_tree already carry v(R) and Delta from
        construction; with skip_if_present we reuse those instead of
        re-querying, so the 2n-1 'evaluation' queries collapse into the
        construction queries. Returns the number of *fresh* queries spent here
        (normally 0).
        """
        all_nodes: list[_Node] = []

        def collect(n: _Node):
            all_nodes.append(n)
            for c in n.children:
                collect(c)

        collect(root)

        n_queries = 0
        for nd in all_nodes:
            if skip_if_present and (nd.v != 0.0 or nd.is_leaf):
                continue  # value already set during construction (no model call)
            nd.v = self._value_of_mask(nd.mask, x, b, x0_val, target)
            n_queries += 1

        for nd in all_nodes:
            if not nd.is_leaf:
                nd.delta = nd.v - sum(c.v for c in nd.children)

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

        f_x = self._target_prob(x, target)
        f_b = self._target_prob(b, target)   # x0 = full-blur reveal of nothing
        x0_val = f_b

        # Build leaf segmentation, then the model-aware agglomerative tree.
        labels = self._leaf_labels(img01)
        root, tinfo = self._build_tree(img01, labels, x, b, x0_val, target)

        # Fill v(R) and Delta(R) across the tree (reusing construction values).
        n_queries = self._compute_values(root, x, b, x0_val, target)

        # --- telescoping identity check ------------------------------------- #
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
        identity_lhs = float(root.v)                  # v(root)
        identity_rhs = sum_leaf_v + sum_delta         # sum leaf v + sum Delta
        identity_residual = identity_lhs - identity_rhs  # ~0 up to float error

        # --- non-additivity index ------------------------------------------- #
        denom = sum(abs(l.v) for l in leaves) + sum(abs(r.delta) for r in internals)
        nai = float(sum(abs(r.delta) for r in internals) / denom) if denom > 0 else 0.0

        # --- per-pixel attribution: leaf-additive density ------------------- #
        attr = np.zeros((H, W), dtype=np.float64)
        for leaf in leaves:
            area = max(leaf.area, 1)
            attr[leaf.mask] = leaf.v / area

        f_phi = self._target_prob(self._phi(x, b, root.mask), target)

        # --- sufficiency diagnostics ---------------------------------------- #
        suff_id = tinfo["suff_node_id"]
        suff_node = next((nd for nd in (leaves + internals) if nd.id == suff_id), root)
        suff_area_frac = float(suff_node.area) / float(H * W)
        resid_above_suff = float(root.v - suff_node.v)
        resid_above_frac = (resid_above_suff / root.v) if root.v != 0 else 0.0

        # --- serialize the tree --------------------------------------------- #
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
                "merge_criterion": self.merge_criterion,
                "frontier_k": self.frontier_k,
                "n_value_queries": n_queries,                 # fresh queries here (~0)
                "n_construction_queries": tinfo["n_construction_queries"],
                "n_total_queries": n_queries + tinfo["n_construction_queries"],
                "root_v": float(root.v),
                "sum_leaf_v": sum_leaf_v,
                "sum_delta": sum_delta,
                "identity_lhs": identity_lhs,
                "identity_rhs": identity_rhs,
                "identity_residual": identity_residual,       # telescoping check ~0
                "nai": nai,                                   # non-additivity index
                # --- merge / sufficiency reporting ---
                "merge_order": tinfo["merge_order"],          # (a,b,merged,delta)
                "merge_deltas": tinfo["merge_deltas"],        # residual per merge
                "suff_node_id": suff_id,
                "suff_v": float(suff_node.v),
                "suff_area_frac": suff_area_frac,             # area(S*)/area(image)
                "resid_above_suff": resid_above_suff,         # v(root)-v(S*)
                "resid_above_frac": resid_above_frac,         # as frac of v(root)
                "suff_eps": self.suff_eps,
                "tree": all_serialized,                       # full v/Delta per node
                "leaf_masks": leaf_masks,
                "reference": "blur_completion",
                "merge_rule": f"agglomerative_{self.merge_criterion}",
            },
        )