"""PyramidExplainer -- nested greedy joint-score chain with merge residuals.

Selection signal
----------------
We build the region tree by *greedily growing a single nested coalition* on the
**joint reveal score** v(S) = f(Phi_S(x)) - f(x0), NOT on the merge residual
Delta. This is the key change from the Delta-greedy construction: Delta is a
*gap* (v(AuB) - v(A) - v(B)); building on Delta is circular (a tree built to
maximise Delta will show large Delta). v(S) is bounded above by v(root) and is
a genuine model output, so optimising it is non-circular -- we optimise the
quantity we report.

Construction (nested greedy-add-leaf)
-------------------------------------
    S_1 = argmax_l v({l})
    S_k = S_{k-1} u {argmax_{l not in S_{k-1}} v(S_{k-1} u {l})}

Each step appends one leaf, creating one binary internal node (running union +
the newly added leaf). The chain S_1 c S_2 c ... c S_n is the trace from a
single leaf up to the root (= all leaves). Because the partition at every node
is disjoint (parent = running-union-child + one fresh-leaf-child), Theorem 1
(telescoping) holds unchanged:

    v(root) = sum_{leaves} v(leaf) + sum_{internal} Delta(R)

No adjacency constraint: the best next leaf may be spatially non-contiguous
(the cooperative Figure-1 case). A lazy-greedy (CELF) upper-bound heap keeps the
forward count near-linear when v is roughly submodular.

Optimality claim (honest)
-------------------------
Greedy yields the *nested-greedy-best* size-k set, exactly optimal iff v is
monotone submodular and (1-1/e)-optimal otherwise -- NOT the global best size-k
set. We strengthen and certify each level with a **1-swap local-search audit**:
try replacing one in-set leaf with one out-of-set leaf; if no single swap raises
v, S_k is a certified 1-swap local optimum. This is the checkable form of "no
other same-size set scores higher" (local / high-probability, since v is not
guaranteed monotone). Swaps may break strict nesting, so the *audit* is reported
separately and the *telescoping tree* is always built from the pure nested chain.

Controls (mandatory, not optional)
----------------------------------
Random-tree / color-tree controls compare how fast v(S_k) rises with k against
the greedy chain. If a blind tree concentrates the model score just as fast, the
chain is not doing real work. Reference (blur sigma) and query counts are logged.

Per-pixel attribution remains the leaf-additive density v(leaf)/area(leaf);
Delta, the sufficiency node, the greedy concentration curve, and the swap audit
are reported in `extras`.
"""
from __future__ import annotations

import heapq
import itertools
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F

try:  # scikit-image is used for the leaf segmentation
    from skimage.segmentation import slic
    _HAS_SKIMAGE = True
except Exception:  # pragma: no cover - skimage layout varies across versions
    _HAS_SKIMAGE = False

from .base import AttributionResult, Explainer, blur_reference, denormalize


@dataclass
class _Node:
    """One region in the region tree."""
    id: int
    mask: np.ndarray            # (H,W) bool, pixels belonging to this region
    children: list = field(default_factory=list)  # list[_Node]; empty for leaves
    v: float = 0.0              # holistic value v(R) = f(Phi_R(x)) - f(x0)
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
        value_mode: str = "prob",        # "logit" (paper Def 3) or "prob"
        swap_audit: bool = True,          # run 1-swap local-optimality certificate
        swap_levels: Optional[list] = None,  # which k to audit; None -> a few checkpoints
        run_controls: bool = True,        # random + color tree concentration controls
        n_random_trees: int = 5,          # random-chain controls to average
        suff_eps: float = 0.05,           # sufficiency: v(S) >= (1-eps) v(root)
        seed: int = 0,
        **kw,
    ):
        super().__init__(*args, **kw)
        if value_mode not in ("logit", "prob"):
            raise ValueError(f"value_mode must be 'logit' or 'prob', got {value_mode!r}")
        self.sigma = sigma
        self.n_segments = n_segments
        self.compactness = compactness
        self.value_mode = value_mode
        self.swap_audit = swap_audit
        self.swap_levels = swap_levels
        self.run_controls = run_controls
        self.n_random_trees = n_random_trees
        self.suff_eps = suff_eps
        self.seed = seed

    # ------------------------------------------------------------------ #
    # Phi: blur-completion projection operator
    # ------------------------------------------------------------------ #
    def _phi(self, x: torch.Tensor, b: torch.Tensor, mask_bool: np.ndarray) -> torch.Tensor:
        """Phi_R(x) = m * x + (1 - m) * b, m the (H,W) {0,1} region indicator."""
        m = torch.as_tensor(mask_bool, dtype=x.dtype, device=x.device)
        m = m.view(1, 1, *mask_bool.shape)  # (1,1,H,W) broadcast over channels
        return m * x + (1.0 - m) * b

    def _score(self, comp: torch.Tensor, target: int) -> float:
        """Raw model response on a completed image: logit (default) or prob."""
        with torch.no_grad():
            out = self.model(comp)
            if self.value_mode == "logit":
                return float(out[:, target].item())
            return float(F.softmax(out, dim=1)[:, target].item())

    def _value_of_mask(self, mask_bool, x, b, x0_val: float, target: int) -> float:
        """v(R) = score(Phi_R(x)) - x0_val."""
        return self._score(self._phi(x, b, mask_bool), target) - x0_val

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
        ys = (np.arange(H) * side // H).astype(np.int64)
        xs = (np.arange(W) * side // W).astype(np.int64)
        labels = (ys[:, None] * side + xs[None, :]).astype(np.int64)
        return labels

    # ------------------------------------------------------------------ #
    # nested greedy-add-leaf chain on the joint score v(union)
    # ------------------------------------------------------------------ #
    def _build_tree(self, img01, labels, x, b, x0_val: float, target: int):
        """Grow one nested coalition S_1 c S_2 c ... by greedily adding the leaf
        that most increases the joint score v(S u {l}). Returns (root, info).

        Lazy-greedy (CELF): each candidate leaf keeps an upper bound on its
        marginal gain (its gain at the last time it was evaluated). The leaf with
        the highest stale bound is re-evaluated against the current set; if its
        fresh gain still tops the heap, it is accepted -- otherwise it is pushed
        back with the refreshed (smaller) bound. Under submodularity this gives
        the exact greedy pick with far fewer forwards than n per step.
        """
        uniq = [int(l) for l in np.unique(labels)]
        leaf_mask = {l: (labels == l) for l in uniq}
        leaf_color = {l: img01[leaf_mask[l]].mean(axis=0) for l in uniq}

        n_fresh_q = 0

        # ---- leaf nodes + individual values --------------------------- #
        node: dict[int, _Node] = {}
        leaf_node: dict[int, _Node] = {}
        leaf_value: dict[int, float] = {}
        next_id = 0
        for l in uniq:
            v_l = self._value_of_mask(leaf_mask[l], x, b, x0_val, target)
            n_fresh_q += 1
            nd = _Node(id=next_id, mask=leaf_mask[l], children=[], v=v_l)
            node[next_id] = nd
            leaf_node[l] = nd
            leaf_value[l] = v_l
            next_id += 1

        # ---- greedy state --------------------------------------------- #
        # S_1: leaf with the largest individual value.
        first = max(uniq, key=lambda l: leaf_value[l])
        cur_mask = leaf_mask[first].copy()
        cur_v = leaf_value[first]
        cur_node = leaf_node[first]            # running-union node (starts as a leaf)
        in_set = {first}

        chain = [first]                         # leaf-label order of inclusion
        chain_v = [cur_v]                       # v(S_k) after each inclusion
        chain_delta = [0.0]                     # Delta at each merge (0 for S_1)
        chain_node_ids = [cur_node.id]          # running-union node id per level
        merge_order = []                        # (running_id, leaf_id, merged_id, delta)

        # lazy-greedy heap of candidate leaves: (-upper_bound_gain, tiebreak, leaf)
        heap: list = []
        tiebreak = itertools.count()
        # initial upper bound on each candidate's marginal gain: gain of adding it
        # to S_1 is unknown, so seed with its own leaf value (a cheap proxy; CELF
        # only requires the bound to start >= true gain in the submodular case,
        # which v({l}) is when v is submodular and monotone -- otherwise CELF still
        # returns a valid greedy pick because we always re-evaluate before accept).
        for l in uniq:
            if l == first:
                continue
            heapq.heappush(heap, (-leaf_value[l], next(tiebreak), l))

        # ---- grow the chain ------------------------------------------- #
        while len(in_set) < len(uniq):
            # CELF: pop the stalest-best candidate, refresh its gain, accept if it
            # still tops the heap; otherwise reinsert with the refreshed bound.
            best_leaf = None
            best_gain = None
            best_union_mask = None
            best_union_v = None
            while heap:
                neg_bound, _, l = heapq.heappop(heap)
                if l in in_set:
                    continue
                union_mask = cur_mask | leaf_mask[l]
                v_union = self._value_of_mask(union_mask, x, b, x0_val, target)
                n_fresh_q += 1
                gain = v_union - cur_v
                # peek the next stale bound; accept if our fresh gain beats it
                nxt = -heap[0][0] if heap else -np.inf
                if gain >= nxt:
                    best_leaf, best_gain = l, gain
                    best_union_mask, best_union_v = union_mask, v_union
                    break
                else:
                    heapq.heappush(heap, (-gain, next(tiebreak), l))

            if best_leaf is None:  # heap drained of valid candidates
                break

            l = best_leaf
            # new internal node: children = running-union node + the new leaf node
            leaf_child = leaf_node[l]
            merged = _Node(
                id=next_id,
                mask=best_union_mask,
                children=[cur_node, leaf_child],
                v=best_union_v,
            )
            # Delta at this merge: v(S_k) - [v(S_{k-1}) + v({l})]
            merged.delta = best_union_v - (cur_v + leaf_value[l])
            node[next_id] = merged
            merge_order.append((cur_node.id, leaf_child.id, next_id, float(merged.delta)))

            # advance state
            cur_node = merged
            cur_mask = best_union_mask
            cur_v = best_union_v
            in_set.add(l)
            next_id += 1

            chain.append(l)
            chain_v.append(cur_v)
            chain_delta.append(float(merged.delta))
            chain_node_ids.append(cur_node.id)

        root = cur_node

        # ---- sufficiency node: smallest-area node with v >= (1-eps) v_root #
        v_root_est = root.v
        thresh = (1.0 - self.suff_eps) * v_root_est if v_root_est > 0 else v_root_est
        suff_node, suff_area = root, root.area
        for nd in node.values():
            if nd.v >= thresh and nd.area < suff_area:
                suff_node, suff_area = nd, nd.area

        info = {
            "chain": chain,                       # leaf labels in inclusion order
            "chain_v": chain_v,                   # v(S_k) per level (concentration curve)
            "chain_delta": chain_delta,           # Delta realised at each merge
            "chain_node_ids": chain_node_ids,
            "merge_order": merge_order,
            "v_root_est": float(v_root_est),
            "suff_node_id": int(suff_node.id),
            "suff_v": float(suff_node.v),
            "n_construction_queries": int(n_fresh_q),
            "leaf_value": leaf_value,
            "leaf_mask": leaf_mask,
            "leaf_color": leaf_color,
            "uniq": uniq,
        }
        return root, info

    # ------------------------------------------------------------------ #
    # 1-swap local-optimality audit
    # ------------------------------------------------------------------ #
    def _swap_audit(self, tinfo, x, b, x0_val, target):
        """For selected levels k, test whether any single in/out swap raises
        v(S_k). Returns per-level certificates and counts fresh forwards used."""
        chain = tinfo["chain"]
        leaf_mask = tinfo["leaf_mask"]
        uniq = tinfo["uniq"]
        n = len(uniq)

        if self.swap_levels is not None:
            levels = [k for k in self.swap_levels if 1 <= k <= len(chain)]
        else:  # default checkpoints: 1, n/8, n/4, n/2 leaves (where structure is read)
            cand = sorted({1, max(1, n // 8), max(1, n // 4), max(1, n // 2)})
            levels = [k for k in cand if k <= len(chain)]

        n_q = 0
        audits = {}
        for k in levels:
            S = set(chain[:k])
            outside = [l for l in uniq if l not in S]
            # base mask / value for S_k
            base_mask = np.zeros_like(next(iter(leaf_mask.values())))
            for l in S:
                base_mask = base_mask | leaf_mask[l]
            v_S = self._value_of_mask(base_mask, x, b, x0_val, target)
            n_q += 1

            best_improve = 0.0
            best_swap = None
            for l_in in list(S):
                for l_out in outside:
                    swapped = (S - {l_in}) | {l_out}
                    m = np.zeros_like(base_mask)
                    for l in swapped:
                        m = m | leaf_mask[l]
                    v_sw = self._value_of_mask(m, x, b, x0_val, target)
                    n_q += 1
                    improve = v_sw - v_S
                    if improve > best_improve:
                        best_improve = improve
                        best_swap = (int(l_in), int(l_out))

            audits[int(k)] = {
                "v_S": float(v_S),
                "is_1swap_local_opt": bool(best_swap is None),
                "best_improving_swap": best_swap,      # None => certified local opt
                "best_improve": float(best_improve),
            }
        return audits, n_q

    # ------------------------------------------------------------------ #
    # control chains: how fast does v(S_k) rise for a blind ordering?
    # ------------------------------------------------------------------ #
    def _control_curves(self, tinfo, x, b, x0_val, target):
        """Random-leaf-order and color-order nested chains, for comparison with
        the greedy concentration curve. Returns mean curves and forward count."""
        uniq = tinfo["uniq"]
        leaf_mask = tinfo["leaf_mask"]
        leaf_color = tinfo["leaf_color"]
        rng = np.random.default_rng(self.seed)
        n_q = 0

        def curve_for_order(order):
            nonlocal n_q
            m = np.zeros_like(next(iter(leaf_mask.values())))
            vs = []
            for l in order:
                m = m | leaf_mask[l]
                vs.append(self._value_of_mask(m, x, b, x0_val, target))
                n_q += 1
            return vs

        # random controls (averaged)
        rand_curves = []
        for _ in range(self.n_random_trees):
            order = list(uniq)
            rng.shuffle(order)
            rand_curves.append(curve_for_order(order))
        rand_mean = np.mean(np.array(rand_curves), axis=0).tolist()

        # color control: order by descending brightness (a model-blind heuristic)
        bright = {l: float(np.mean(leaf_color[l])) for l in uniq}
        color_order = sorted(uniq, key=lambda l: bright[l], reverse=True)
        color_curve = curve_for_order(color_order)

        return {
            "random_mean_curve": rand_mean,
            "color_curve": color_curve,
            "n_random_trees": self.n_random_trees,
        }, n_q

    # ------------------------------------------------------------------ #
    # telescoping value/residual fill (reuses construction values)
    # ------------------------------------------------------------------ #
    def _collect(self, root: _Node):
        leaves, internals, alln = [], [], []

        def rec(nd):
            alln.append(nd)
            (leaves if nd.is_leaf else internals).append(nd)
            for c in nd.children:
                rec(c)

        rec(root)
        return leaves, internals, alln

    # ------------------------------------------------------------------ #
    # explain
    # ------------------------------------------------------------------ #
    def explain(self, x: torch.Tensor) -> AttributionResult:
        x = x.to(self.device)
        target = self._resolve_target(x)

        b = blur_reference(x, self.sigma).to(self.device)
        img01 = denormalize(x)[0].permute(1, 2, 0).cpu().numpy()  # (H,W,3) in [0,1]
        H, W = img01.shape[:2]

        f_x = self._score(x, target)
        f_b = self._score(b, target)
        x0_val = f_b

        labels = self._leaf_labels(img01)
        root, tinfo = self._build_tree(img01, labels, x, b, x0_val, target)

        leaves, internals, _ = self._collect(root)
        leaf_masks = {leaf.id: leaf.mask for leaf in leaves}

        # --- telescoping identity (Delta already set at construction) ------- #
        sum_leaf_v = float(sum(l.v for l in leaves))
        sum_delta = float(sum(r.delta for r in internals))
        identity_lhs = float(root.v)
        identity_rhs = sum_leaf_v + sum_delta
        identity_residual = identity_lhs - identity_rhs   # ~0 up to float error

        # --- non-additivity index ------------------------------------------ #
        denom = sum(abs(l.v) for l in leaves) + sum(abs(r.delta) for r in internals)
        nai = float(sum(abs(r.delta) for r in internals) / denom) if denom > 0 else 0.0

        # --- per-pixel attribution: leaf-additive density ------------------- #
        attr = np.zeros((H, W), dtype=np.float64)
        for leaf in leaves:
            attr[leaf.mask] = leaf.v / max(leaf.area, 1)

        f_phi = self._score(self._phi(x, b, root.mask), target)

        # --- sufficiency diagnostics ---------------------------------------- #
        suff_id = tinfo["suff_node_id"]
        suff_node = next((nd for nd in (leaves + internals) if nd.id == suff_id), root)
        suff_area_frac = float(suff_node.area) / float(H * W)
        resid_above_suff = float(root.v - suff_node.v)
        resid_above_frac = (resid_above_suff / root.v) if root.v != 0 else 0.0

        # --- optional 1-swap audit and controls ----------------------------- #
        n_audit_q = 0
        swap_audits = None
        if self.swap_audit:
            swap_audits, n_audit_q = self._swap_audit(tinfo, x, b, x0_val, target)

        n_control_q = 0
        controls = None
        if self.run_controls:
            controls, n_control_q = self._control_curves(tinfo, x, b, x0_val, target)

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

        n_construction_q = tinfo["n_construction_queries"]
        return AttributionResult(
            attribution=attr,
            method=self.name,
            target_class=target,
            target_class_name=self._class_name(target),
            f_x=f_x,
            f_b=f_b,
            f_phi=f_phi,
            extras={
                "value_mode": self.value_mode,
                "sigma": self.sigma,
                "n_segments": self.n_segments,
                "n_leaves": len(leaves),
                "n_internal": len(internals),
                "merge_rule": "nested_greedy_joint_score",
                # query accounting
                "n_construction_queries": n_construction_q,
                "n_audit_queries": n_audit_q,
                "n_control_queries": n_control_q,
                "n_total_queries": n_construction_q + n_audit_q + n_control_q,
                # telescoping self-check
                "root_v": float(root.v),
                "sum_leaf_v": sum_leaf_v,
                "sum_delta": sum_delta,
                "identity_lhs": identity_lhs,
                "identity_rhs": identity_rhs,
                "identity_residual": identity_residual,
                "nai": nai,
                # the nested greedy trace (the main object)
                "chain": tinfo["chain"],
                "chain_v": tinfo["chain_v"],            # greedy concentration curve
                "chain_delta": tinfo["chain_delta"],
                "chain_node_ids": tinfo["chain_node_ids"],
                "merge_order": tinfo["merge_order"],
                # sufficiency
                "suff_node_id": suff_id,
                "suff_v": float(suff_node.v),
                "suff_area_frac": suff_area_frac,
                "resid_above_suff": resid_above_suff,
                "resid_above_frac": resid_above_frac,
                "suff_eps": self.suff_eps,
                # optimality audit + controls
                "swap_audits": swap_audits,             # per-level 1-swap certificates
                "controls": controls,                   # random/color concentration curves
                # full tree + leaf masks
                "tree": all_serialized,
                "leaf_masks": leaf_masks,
                "reference": "blur_completion",
            },
        )