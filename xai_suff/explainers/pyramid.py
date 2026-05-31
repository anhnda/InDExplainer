"""PyramidExplainer -- hierarchical on-manifold attribution with a MODEL-AWARE,
marginal-evidence merge objective.

Built on the simple, working version (softmax class score in [0,1], min 0;
blur-completion reveal value; telescoping identity). The ONLY change is the tree
construction: instead of merging by mean color (a heuristic with no model-level
justification), we merge by an explicit, stated, model-aware objective.

Why color-merge was wrong
-------------------------
Color linkage builds the tree without ever consulting the model, so the regions
it groups need not correspond to what the model relies on. The tree is then an
appearance artifact, not an explanation.

Why "value gain over the blur baseline" was ALSO wrong
------------------------------------------------------
A naive model-aware rule scores a merge by v(u|v) = f(Phi_{u|v}) - f(b), i.e.
how much revealing the union raises the score above the fully-blurred reference.
But f(b) ~ 0 means *every* sharp reveal -- sky included -- raises the score
simply by un-blurring the image. That rule cannot tell object evidence from
"the image is no longer blurred", so it grows the background. Confirmed
empirically: it produced large redundant sky/grass nodes, not a church focus.

The objective we actually use: marginal evidence (deletion effect)
------------------------------------------------------------------
A region is evidence to the extent that REMOVING it from the full sharp image
hurts the target score:

    m(R) = f(x) - f( Phi_{R^c}(x) )            # blur out R, keep the rest sharp

    sky : remove it, church still recognized -> f(Phi_{R^c}) ~ f(x) -> m ~ 0
    church: remove it, recognition collapses  -> f(Phi_{R^c}) << f(x) -> m large

This is offset-free (there is no blur baseline to climb, so un-blurring cannot be
gamed) and causal (it is a deletion effect on the true prediction). It is the
right model-aware signal for "what does the model actually use".

Merge rule (greedy, stated objective). At each step merge the adjacent pair
whose UNION carries the most marginal evidence per pixel:

    score(u,v) = m(u|v) / area(u|v) ,    (u*,v*) = argmax_{(u,v) in adj} score

High-evidence (object) regions therefore assemble FIRST into a compact, high-
value superregion; low-evidence (background) regions stay scattered and merge
late. The object appears as a real internal node R* with v(R*) ~= v(root), and
the final root merge (object + background) adds little -- exactly the "focal
superregion takes the score, not the root remainder" behaviour required.

Blind proposal, model-aware selection (cost control). A pure greedy max would
need a deletion query for every adjacent pair at every level. We shortlist the k
most backbone-feature-similar adjacent pairs (model-BLIND), then evaluate the
marginal-evidence objective only on those k. -> O(n*k) forward passes.

Node value and identity are UNCHANGED from the simple version
-------------------------------------------------------------
Reported node value is still the holistic reveal value
    v(R) = f(Phi_R(x)) - f(b),   f = softmax class score in [0,1] (min 0)
and the telescoping identity
    v(root) = sum_leaves v(leaf) + sum_internal Delta(R)
still holds exactly (it holds for ANY tree). Marginal evidence m(R) is used ONLY
to steer the merge; it is the construction objective, not the reported value. So
localization is model-aware and principled, while the value accounting and its
self-check are preserved.

Focal node: R* = minimal-area node with v(R*) >= tau * v(root). Its subtree is
the model's evidence; everything above it is remainder. The per-pixel map is the
focal-subtree leaf density.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F

try:
    from skimage.segmentation import slic
    _HAS_SKIMAGE = True
except Exception:  # pragma: no cover
    _HAS_SKIMAGE = False

from .base import AttributionResult, Explainer, blur_reference, denormalize


@dataclass
class _Node:
    id: int
    mask: np.ndarray            # (H,W) bool
    children: list              # list[_Node]; empty for leaves
    v: float = 0.0              # holistic reveal value f(Phi_R) - f(b)
    delta: float = 0.0          # v(R) - sum_j v(child_j); 0 for leaves
    m: float = 0.0              # marginal evidence f(x) - f(Phi_{R^c}); steering only

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
        sigma: float = 11.0,
        n_segments: int = 144,
        compactness: float = 2.0,
        # --- model-aware merge objective controls --- #
        shortlist_k: int = 6,           # blind-proposal shortlist size per step
        feature_layer: str = "layer3",  # backbone layer for blind shortlist
        merge_mode: str = "evidence",   # "evidence" (marginal) | "color" (legacy)
        evidence_area_pow: float = 1.0, # area exponent in score = m / area^pow
        focal_tau: float = 0.8,         # focal node: v(R*) >= tau * v(root)
        **kw,
    ):
        super().__init__(*args, **kw)
        self.sigma = sigma
        self.n_segments = n_segments
        self.compactness = compactness
        self.shortlist_k = shortlist_k
        self.feature_layer = feature_layer
        self.merge_mode = merge_mode
        self.evidence_area_pow = evidence_area_pow
        self.focal_tau = focal_tau
        self._build_queries = 0

    # ------------------------------------------------------------------ #
    # Phi + score (softmax class score in [0,1], min 0 -- no logits)
    # ------------------------------------------------------------------ #
    def _phi(self, x, b, mask_bool):
        m = torch.as_tensor(mask_bool, dtype=x.dtype, device=x.device)
        m = m.view(1, 1, *mask_bool.shape)
        return m * x + (1.0 - m) * b

    def _phi_complement(self, x, b, mask_bool):
        """Reveal everything EXCEPT R sharp; blur out R. (For deletion / m(R).)"""
        m = torch.as_tensor(mask_bool, dtype=x.dtype, device=x.device)
        m = m.view(1, 1, *mask_bool.shape)
        return (1.0 - m) * x + m * b

    def _target_prob(self, comp, target: int) -> float:
        with torch.no_grad():
            return float(F.softmax(self.model(comp), dim=1)[:, target].item())

    # --- holistic reveal value (reported; identity uses this) --- #
    def _value(self, x, b, mask_bool, x0_val: float, target: int) -> float:
        self._build_queries += 1
        return self._target_prob(self._phi(x, b, mask_bool), target) - x0_val

    # --- marginal evidence (steering only): how much removing R costs --- #
    def _marginal(self, x, b, mask_bool, fx: float, target: int) -> float:
        self._build_queries += 1
        return fx - self._target_prob(self._phi_complement(x, b, mask_bool), target)

    # ------------------------------------------------------------------ #
    def _leaf_labels(self, img01):
        if _HAS_SKIMAGE:
            return slic(img01, n_segments=self.n_segments,
                        compactness=self.compactness, start_label=0,
                        channel_axis=2).astype(np.int64)
        H, W = img01.shape[:2]
        side = max(1, int(round(np.sqrt(self.n_segments))))
        ys = np.linspace(0, side, H, endpoint=False).astype(np.int64)
        xs = np.linspace(0, side, W, endpoint=False).astype(np.int64)
        return (ys[:, None] * side + xs[None, :]).astype(np.int64)

    def _feature_map(self, x, H, W):
        module = dict(self.model.named_modules()).get(self.feature_layer)
        if module is None:
            return None
        feats = {}
        def hook(_m, _i, out): feats["a"] = out.detach()
        h = module.register_forward_hook(hook)
        try:
            with torch.no_grad():
                self.model(x)
        finally:
            h.remove()
        a = feats.get("a")
        if a is None:
            return None
        a = F.interpolate(a, size=(H, W), mode="bilinear", align_corners=False)
        return a[0].permute(1, 2, 0).cpu().numpy().astype(np.float64)

    @staticmethod
    def _descriptor(field, mask):
        return field[mask].mean(axis=0)

    # ------------------------------------------------------------------ #
    # model-aware tree: greedy max of marginal-evidence-per-pixel
    # ------------------------------------------------------------------ #
    def _build_tree(self, labels, x, b, x0_val, fx, target, desc_field):
        H, W = labels.shape
        next_id = 0
        nodes: dict[int, _Node] = {}
        for lab in np.unique(labels):
            mask = labels == lab
            nd = _Node(id=next_id, mask=mask, children=[])
            nd.v = self._value(x, b, mask, x0_val, target)   # cached reveal value
            nodes[next_id] = nd
            next_id += 1
        active = set(nodes.keys())
        desc = {nid: self._descriptor(desc_field, nodes[nid].mask) for nid in active}

        def adjacency(masks):
            owner = -np.ones((H, W), dtype=np.int64)
            for nid, m in masks.items():
                owner[m] = nid
            pairs = set()
            a, c = owner[:, :-1], owner[:, 1:]
            for u, v in zip(a[a != c], c[a != c]):
                if u >= 0 and v >= 0:
                    pairs.add((min(int(u), int(v)), max(int(u), int(v))))
            a, c = owner[:-1, :], owner[1:, :]
            for u, v in zip(a[a != c], c[a != c]):
                if u >= 0 and v >= 0:
                    pairs.add((min(int(u), int(v)), max(int(u), int(v))))
            return pairs

        adj = adjacency({nid: nodes[nid].mask for nid in active})

        def blind_dist(p):
            return float(np.linalg.norm(desc[p[0]] - desc[p[1]]))

        def score(m_uv, area):
            return m_uv / max(float(area) ** self.evidence_area_pow, 1.0)

        while len(active) > 1:
            if not adj:
                acts = list(active)
                cents = {n: np.argwhere(nodes[n].mask).mean(axis=0) for n in acts}
                best, bd = None, np.inf
                for i in range(len(acts)):
                    for j in range(i + 1, len(acts)):
                        d = float(np.linalg.norm(cents[acts[i]] - cents[acts[j]]))
                        if d < bd:
                            bd, best = d, (acts[i], acts[j])
                u, v = best
                mm = nodes[u].mask | nodes[v].mask
                v_uv = self._value(x, b, mm, x0_val, target)
                m_uv = self._marginal(x, b, mm, fx, target)
            elif self.merge_mode == "color":
                u, v = min(adj, key=blind_dist)
                mm = nodes[u].mask | nodes[v].mask
                v_uv = self._value(x, b, mm, x0_val, target)
                m_uv = self._marginal(x, b, mm, fx, target)
            else:
                # evidence mode: blind shortlist -> model-aware argmax of m/area
                shortlist = sorted(adj, key=blind_dist)[: max(1, self.shortlist_k)]
                best, best_s, best_v, best_m = None, -np.inf, 0.0, 0.0
                for (p, q) in shortlist:
                    mm = nodes[p].mask | nodes[q].mask
                    a = int(mm.sum())
                    m_pq = self._marginal(x, b, mm, fx, target)
                    s = score(m_pq, a)
                    if s > best_s:
                        best_s, best, best_m = s, (p, q), m_pq
                        best_v = self._value(x, b, mm, x0_val, target)
                u, v = best
                mm = nodes[u].mask | nodes[v].mask
                v_uv, m_uv = best_v, best_m

            merged = _Node(id=next_id, mask=mm, children=[nodes[u], nodes[v]],
                           v=v_uv, m=m_uv)
            nodes[next_id] = merged
            desc[next_id] = self._descriptor(desc_field, mm)
            active.discard(u); active.discard(v); active.add(next_id)
            new_adj = set()
            for (p, q) in adj:
                if p in (u, v): p = next_id
                if q in (u, v): q = next_id
                if p != q and p in active and q in active:
                    new_adj.add((min(p, q), max(p, q)))
            adj = new_adj
            next_id += 1

        return nodes[next(iter(active))]

    def _fill_deltas(self, root):
        st = [root]
        while st:
            n = st.pop()
            if not n.is_leaf:
                n.delta = n.v - sum(c.v for c in n.children)
                st.extend(n.children)

    @staticmethod
    def _find_focal(root, tau):
        tv = tau * root.v
        best, best_area = root, root.area
        st = [root]
        while st:
            n = st.pop()
            if n.v >= tv and n.area <= best_area:
                best, best_area = n, n.area
            st.extend(n.children)
        return best

    # ------------------------------------------------------------------ #
    def explain(self, x: torch.Tensor) -> AttributionResult:
        x = x.to(self.device)
        target = self._resolve_target(x)
        self._build_queries = 0

        b = blur_reference(x, self.sigma).to(self.device)
        img01 = denormalize(x)[0].permute(1, 2, 0).cpu().numpy()
        H, W = img01.shape[:2]

        f_x = self._target_prob(x, target)
        f_b = self._target_prob(b, target)
        x0_val = f_b

        labels = self._leaf_labels(img01)

        desc_field = None
        if self.merge_mode != "color":
            desc_field = self._feature_map(x, H, W)
        feature_used = desc_field is not None
        if desc_field is None:
            desc_field = img01

        root = self._build_tree(labels, x, b, x0_val, f_x, target, desc_field)
        self._fill_deltas(root)

        leaves, internals = [], []
        def split(n):
            (leaves if n.is_leaf else internals).append(n)
            for c in n.children: split(c)
        split(root)
        leaf_masks = {l.id: l.mask for l in leaves}

        focal = self._find_focal(root, self.focal_tau)
        focal_leaves = []
        def cfl(n):
            if n.is_leaf: focal_leaves.append(n)
            for c in n.children: cfl(c)
        cfl(focal)

        sum_leaf_v = float(sum(l.v for l in leaves))
        sum_delta = float(sum(r.delta for r in internals))
        identity_lhs = float(root.v)
        identity_rhs = sum_leaf_v + sum_delta
        identity_residual = identity_lhs - identity_rhs

        root_v = float(root.v) if abs(root.v) > 1e-12 else 1e-12
        focal_fraction = float(focal.v) / root_v
        focal_area_frac = focal.area / float(H * W)
        root_delta_frac = float(root.delta) / root_v

        attr = np.zeros((H, W), dtype=np.float64)
        for l in focal_leaves:
            attr[l.mask] = l.v / max(l.area, 1)
        attr_all = np.zeros((H, W), dtype=np.float64)
        for l in leaves:
            attr_all[l.mask] = l.v / max(l.area, 1)

        f_phi = self._target_prob(self._phi(x, b, root.mask), target)

        def ser(n):
            return {"id": n.id, "area": n.area, "v": float(n.v),
                    "delta": float(n.delta), "m": float(n.m),
                    "is_leaf": n.is_leaf, "child_ids": [c.id for c in n.children],
                    "is_focal": n.id == focal.id}
        all_ser = []
        def walk(n):
            all_ser.append(ser(n))
            for c in n.children: walk(c)
        walk(root)

        n_leaves = len(leaves)
        return AttributionResult(
            attribution=attr, method=self.name, target_class=target,
            target_class_name=self._class_name(target),
            f_x=f_x, f_b=f_b, f_phi=f_phi,
            extras={
                "sigma": self.sigma, "n_segments": self.n_segments,
                "n_leaves": n_leaves, "n_internal": len(internals),
                "value_space": "prob_min0",
                "n_value_queries": self._build_queries,
                "merge_mode": self.merge_mode, "shortlist_k": self.shortlist_k,
                "feature_layer": self.feature_layer if feature_used else None,
                "blind_signal": "feature" if feature_used else "color",
                "evidence_area_pow": self.evidence_area_pow,
                "objective": "marginal_evidence_per_pixel",
                "focal_tau": self.focal_tau, "focal_id": focal.id,
                "focal_v": float(focal.v), "focal_area": focal.area,
                "focal_area_frac": focal_area_frac,   # want SMALL
                "focal_fraction": focal_fraction,     # v(R*)/v(root), want >= tau
                "focal_n_leaves": len(focal_leaves),
                "root_v": root_v, "root_delta": float(root.delta),
                "root_delta_frac": root_delta_frac,   # want ~0
                "sum_leaf_v": sum_leaf_v, "sum_delta": sum_delta,
                "identity_lhs": identity_lhs, "identity_rhs": identity_rhs,
                "identity_residual": identity_residual,
                "attr_all_leaves": attr_all,
                "tree": all_ser, "leaf_masks": leaf_masks,
                "reference": "blur_completion",
            },
        )