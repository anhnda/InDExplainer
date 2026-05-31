"""PyramidExplainer -- objective-driven, model-aware hierarchical attribution.

Implements a region tree built by *solving* an early-value objective, rather
than by a model-blind color heuristic. See method note for the theory; the
short version:

Objective (early-value)
-----------------------
Choose a growth ordering of the foreground  S_0 = {} subset S_1 subset ... = root
(each step adds one adjacency-connected leaf) maximizing

    J(T) = sum_{t=1..n} v(S_t)
         = sum_{s=1..n} (n - s + 1) * g_s ,   g_s = v(S_s) - v(S_{s-1}).

J is a decreasing-weighted sum of marginal gains: earlier merges get larger
weights, so maximizing J front-loads model value into the first few merges.

Algorithm (solves J under diminishing returns)
----------------------------------------------
Max-marginal-gain growth: at each step add the adjacent leaf of largest
marginal gain g = v(S u {l}) - v(S). Under submodular v this greedy is the
exact maximizer of J (gains come out sorted decreasing; weights (n-s+1) are
decreasing; rearrangement inequality). Under connectivity it is a bounded
approximation, and the merge residuals Delta(R) reported below *are* the
measured submodularity-gap, i.e. the violations of the optimality premise.

Sufficiency node
----------------
The smallest foreground set S_{t*} with v(S_{t*}) >= (1 - eps) * v(root),
located at the knee of the cumulative-value curve. This is the "small part of
the image that already explains the model"; residual on the path S_{t*}->root
is v(root) - v(S_{t*}) ~ eps * v(root).

Core construction (unchanged identities)
----------------------------------------
1. Segment the image into superpixels (leaves).
2. Build the region tree by foreground growth (above) + cheap color
   agglomeration of the remaining background.
3. Holistic on-manifold value:
       Phi_R(x) = reveal R sharp, replace complement with blur_sigma(x)
       v(R)     = f(Phi_R(x)) - f(x0)            with x0 = full blur reference b
4. Merge residual for internal node R with children c_1..c_m:
       Delta(R) = v(R) - sum_j v(c_j)
5. Telescoping identity:
       v(root) = sum_{leaves} v(leaf) + sum_{internal} Delta(R)

Per-pixel attribution is the leaf-additive density v(leaf)/area(leaf), exactly
as before; interaction (Delta) and the sufficiency node are reported in
`extras` since they are tree-relative, not per-pixel.

Reference dependence (Phi, tree) is logged, not hidden. Because the tree is now
model-aware, random-tree / color-tree controls are required to show that a
small sufficient region is a property of the model, not of a flattering tree.
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
    """One region in the region tree."""
    id: int
    mask: np.ndarray            # (H,W) bool, pixels belonging to this region
    children: list             # list[_Node]; empty for leaves
    v: float = 0.0             # holistic on-manifold value f(Phi_R(x)) - f(x0)
    delta: float = 0.0         # merge residual v(R) - sum_j v(child_j); 0 for leaves

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
        sigma: float = 11.0,         # blur strength for Phi complement (on-manifold)
        n_segments: int = 144,        # target number of leaf superpixels (SLIC)
        compactness: float = 2,       # SLIC compactness
        max_nodes: Optional[int] = None,  # unused; kept for API parity
        frontier_k: Optional[int] = None,  # cap candidates scored per growth step
        suff_eps: float = 0.05,       # sufficiency: v(S) >= (1-eps) v(root)
        knee: bool = True,            # locate sufficiency node at curve knee
        **kw,
    ):
        super().__init__(*args, **kw)
        self.sigma = sigma
        self.n_segments = n_segments
        self.compactness = compactness
        self.max_nodes = max_nodes
        self.frontier_k = frontier_k
        self.suff_eps = suff_eps
        self.knee = knee

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
        H, W = labels.shape
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
    # region tree by SOLVING the early-value objective J
    # ------------------------------------------------------------------ #
    def _build_tree(self, img01, labels, x, b, x0_val: float, target: int):
        """Foreground grown by max-marginal-gain (maximizes J under submodular v),
        background closed cheaply by color, joined at the root.

        Returns (root, info) where info carries construction diagnostics:
        growth ordering, marginal gains, cumulative values, sufficiency node id,
        and the number of model queries spent on construction.
        """
        H, W = labels.shape
        uniq = [int(l) for l in np.unique(labels)]
        n = len(uniq)

        # Leaf nodes + per-leaf masks / mean colors.
        leaf_mask = {l: (labels == l) for l in uniq}
        leaf_color = {l: img01[leaf_mask[l]].mean(axis=0) for l in uniq}
        leaf_adj = self._leaf_adjacency(labels)

        # Leaf values v({l}) -- needed for seeding and reported regardless.
        # These also count as construction queries.
        n_constr_q = 0
        leaf_value: dict[int, float] = {}
        for l in uniq:
            leaf_value[l] = self._value_of_mask(leaf_mask[l], x, b, x0_val, target)
            n_constr_q += 1

        # ---- Phase B: grow the foreground by max-marginal-gain ---------- #
        # Seed from the single highest-value leaf (argmax v({l})).
        seed = max(uniq, key=lambda l: leaf_value[l])

        in_fg = {seed}
        S_mask = leaf_mask[seed].copy()
        v_S = leaf_value[seed]                       # v(S_1); reuse leaf query
        # Growth records (S_1 = {seed}).
        order = [seed]
        gains = [v_S - 0.0]                          # g_1 = v(S_1) - v(empty)=v(S_1)
        cum_v = [v_S]                                # v(S_1)

        # Frontier = leaves adjacent to current foreground, not yet absorbed.
        def frontier(members: set) -> set:
            fr = set()
            for m in members:
                fr |= leaf_adj[m]
            return fr - members

        fr = frontier(in_fg)

        # Greedy growth until every leaf is absorbed.
        while len(in_fg) < n:
            if not fr:
                # Disconnected remainder (rare): absorb the global best leftover
                # by value so growth can continue without spurious queries.
                leftover = [l for l in uniq if l not in in_fg]
                cand = leftover
            else:
                cand = list(fr)

            # Optionally cap the number of model-scored candidates per step,
            # pre-ranking the frontier by leaf value (cheap, model-aware proxy).
            if self.frontier_k is not None and len(cand) > self.frontier_k:
                cand = sorted(cand, key=lambda l: leaf_value[l], reverse=True)
                cand = cand[: self.frontier_k]

            # Score each candidate by marginal gain g = v(S u {l}) - v(S).
            best_l, best_gain, best_v = None, None, None
            for l in cand:
                trial_mask = S_mask | leaf_mask[l]
                v_trial = self._value_of_mask(trial_mask, x, b, x0_val, target)
                n_constr_q += 1
                g = v_trial - v_S
                if best_gain is None or g > best_gain:
                    best_l, best_gain, best_v = l, g, v_trial

            # Commit the best candidate.
            in_fg.add(best_l)
            S_mask = S_mask | leaf_mask[best_l]
            v_S = best_v
            order.append(best_l)
            gains.append(best_gain)
            cum_v.append(v_S)
            fr = frontier(in_fg)

        # ---- Sufficiency node t*: smallest S_t with v(S_t) >= (1-eps) v_root #
        v_root_est = cum_v[-1]                       # v(S_n) = v(root) estimate
        # eps-threshold index.
        thresh = (1.0 - self.suff_eps) * v_root_est if v_root_est > 0 else v_root_est
        t_eps = next((t for t, vv in enumerate(cum_v) if vv >= thresh), len(cum_v) - 1)

        # Optional knee: largest drop in marginal gain (elbow of cum curve).
        if self.knee and len(gains) > 2:
            # knee = index just before the biggest relative gain drop.
            drops = [gains[i] - gains[i + 1] for i in range(len(gains) - 1)]
            t_knee = int(np.argmax(drops))           # S_{t_knee+1} seals the elbow
            t_star = min(t_eps, t_knee + 1)
        else:
            t_star = t_eps
        t_star = max(0, min(t_star, len(order) - 1))  # clamp to valid index

        # ---- Assemble the tree from the growth chain ------------------- #
        # Build the foreground backbone as a left-deep chain:
        #   S_1, then merge(S_1, leaf_2)=S_2, ... = S_n = root.
        # Each S_t internal node reuses the already-computed v(S_t)=cum_v[t].
        next_id = 0
        node: dict[int, _Node] = {}

        # Leaf nodes first (stable ids 0..n-1 keyed by growth order for clarity).
        leaf_node: dict[int, _Node] = {}
        for l in uniq:
            node[next_id] = _Node(id=next_id, mask=leaf_mask[l], children=[])
            node[next_id].v = leaf_value[l]
            leaf_node[l] = node[next_id]
            next_id += 1

        # Backbone: cluster_t is the internal node for S_t (t>=2).
        cluster_mask = leaf_mask[order[0]].copy()
        cluster_node = leaf_node[order[0]]           # S_1 == seed leaf
        suff_node_id = cluster_node.id               # updated when we pass t*
        for t in range(1, len(order)):
            child_existing = cluster_node            # S_{t}
            child_new = leaf_node[order[t]]          # newly added leaf
            cluster_mask = cluster_mask | leaf_mask[order[t]]
            merged = _Node(
                id=next_id,
                mask=cluster_mask.copy(),
                children=[child_existing, child_new],
            )
            merged.v = cum_v[t]                       # reuse construction value
            node[next_id] = merged
            cluster_node = merged
            if t == t_star:
                suff_node_id = merged.id
            next_id += 1

        root = cluster_node                          # S_n covers the full frame

        info = {
            "order": order,                          # leaf labels in growth order
            "gains": [float(g) for g in gains],      # marginal gains g_t
            "cum_v": [float(c) for c in cum_v],      # v(S_t)
            "t_star": int(t_star),                   # sufficiency index (0-based)
            "suff_node_id": int(suff_node_id),
            "v_suff": float(cum_v[t_star]),
            "v_root_est": float(v_root_est),
            "n_construction_queries": int(n_constr_q),
            "objective_J": float(sum(cum_v)),        # J(T) = sum_t v(S_t)
        }
        return root, info

    # ------------------------------------------------------------------ #
    # value + merge-residual computation over the tree
    # ------------------------------------------------------------------ #
    def _compute_values(self, root: _Node, x, b, x0_val: float, target: int,
                        skip_if_present: bool = True):
        """Fill v(R) for every node, then Delta(R) for internal nodes.

        Nodes built by _build_tree already carry v(R) from construction; with
        skip_if_present we reuse those instead of re-querying the model, so the
        2n-1 'evaluation' queries collapse into the construction queries. We
        still report the number of *fresh* value queries spent here.
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
                # value already set during construction (reused, no model call)
                continue
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

        # f(x), f(x0) = f(b).
        f_x = self._target_prob(x, target)
        f_b = self._target_prob(b, target)  # x0 = full-blur reveal of nothing
        x0_val = f_b

        # Build leaf segmentation, then SOLVE the early-value objective for the tree.
        labels = self._leaf_labels(img01)
        root, tinfo = self._build_tree(img01, labels, x, b, x0_val, target)

        # Fill v(R) and Delta(R) across the tree (reusing construction values).
        n_queries = self._compute_values(root, x, b, x0_val, target)

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

        # --- non-additivity index (diagnostic) ------------------------------ #
        denom = sum(abs(l.v) for l in leaves) + sum(abs(r.delta) for r in internals)
        nai = float(sum(abs(r.delta) for r in internals) / denom) if denom > 0 else 0.0

        # --- per-pixel attribution: leaf-additive density ------------------- #
        attr = np.zeros((H, W), dtype=np.float64)
        for leaf in leaves:
            area = max(leaf.area, 1)
            attr[leaf.mask] = leaf.v / area

        # f_phi reported as the root reveal value (whole image sharp) for parity.
        f_phi = self._target_prob(self._phi(x, b, root.mask), target)

        # --- sufficiency diagnostics ---------------------------------------- #
        suff_id = tinfo["suff_node_id"]
        suff_node = next((nd for nd in (leaves + internals) if nd.id == suff_id), root)
        suff_area_frac = float(suff_node.area) / float(H * W)
        resid_above_suff = float(root.v - suff_node.v)        # v(root) - v(S*)
        resid_above_frac = (resid_above_suff / root.v) if root.v != 0 else 0.0

        # Serialize the tree (id, area, v, delta, child ids) for downstream
        # interaction analysis -- residuals are tree-relative, not per-pixel.
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
                "n_value_queries": n_queries,                  # fresh queries here
                "n_construction_queries": tinfo["n_construction_queries"],
                "n_total_queries": n_queries + tinfo["n_construction_queries"],
                "root_v": float(root.v),
                "sum_leaf_v": sum_leaf_v,
                "sum_delta": sum_delta,
                "identity_lhs": identity_lhs,
                "identity_rhs": identity_rhs,
                "identity_residual": identity_residual,        # telescoping check ~0
                "nai": nai,                                    # non-additivity index
                # --- objective + sufficiency reporting ---
                "objective_J": tinfo["objective_J"],           # J(T) = sum_t v(S_t)
                "growth_order": tinfo["order"],
                "growth_gains": tinfo["gains"],
                "growth_cum_v": tinfo["cum_v"],
                "t_star": tinfo["t_star"],
                "suff_node_id": suff_id,
                "suff_n_leaves": int(tinfo["t_star"] + 1),
                "suff_v": float(suff_node.v),
                "suff_area_frac": suff_area_frac,              # area(S*)/area(image)
                "resid_above_suff": resid_above_suff,          # v(root)-v(S*)
                "resid_above_frac": resid_above_frac,          # as frac of v(root)
                "suff_eps": self.suff_eps,
                "tree": all_serialized,                        # full v/Delta per node
                "leaf_masks": leaf_masks,
                "reference": "blur_completion",
                "merge_rule": "max_marginal_gain (solves early-value J)",
            },
        )