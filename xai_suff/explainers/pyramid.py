"""PyramidExplainer -- LIME-primed, sequential nested greedy joint-score chain.

BATCHED VERSION.
================
Algorithm is identical to the original; only the model-forward plumbing changed.
All candidate evaluations that are mutually independent (LIME perturbations, the
per-step greedy candidate set, the swap-audit in/out pairs, and the nested
control chains) are now stacked into a single (N,C,H,W) forward pass, chunked by
`max_batch`, with one GPU->CPU sync per chunk instead of one per image.

What changed vs. the pure-greedy version (unchanged design notes)
-----------------------------------------------------------------
The nested coalition is *seeded* by a LIME prior and then grown by greedy
joint-score steps:

    1. LIME pass on the leaves (SLIC superpixels or grid cells) gives one
       surrogate coefficient per leaf -> a ranked leaf list.
    2. The top-`k_lime` LIME leaves (default k=10) are inserted FIRST, in LIME
       rank order. These form S_1 c S_2 c ... c S_k as the chain seed.
    3. From S_k onward the chain grows by the original nested greedy-add-leaf
       rule on the joint reveal score:
            S_{j} = S_{j-1} u {argmax_{l not in S_{j-1}} v(S_{j-1} u {l})}.

So LIME only fixes the FIRST k leaves; every later leaf is chosen by the real
model joint score v(S u {l}) (the target-class response on the revealed set).

full_lime mode (ablation)
-------------------------
Setting `full_lime=True` makes the ENTIRE chain follow the LIME ranking: the
greedy phase is skipped and the node order is EXACTLY the sorted LIME
coefficients, end-to-end. Theorem 1 (telescoping) still holds, because the tree
is still a disjoint nested partition -- only the ORDER in which leaves are
absorbed changes.

Note on greedy implementation
-----------------------------
The original used CELF lazy-greedy (a per-element upper-bound heap) to keep the
greedy phase near-linear in serial, single-image evaluation. With batching the
bottleneck is no longer the number of v() calls but the number of *forward
passes*; evaluating all remaining candidates for a step in ONE batched forward
turns each step into a single GPU call. So this version does an exact batched
full-evaluation per step (O(n) batched forwards total). The chosen leaf at every
step is the true argmax of v(S u {l}), identical to what CELF converges to, so
the resulting chain is unchanged.

Selection signal, construction, optimality claim, controls, and per-pixel
attribution are all unchanged from the original; see method docstrings.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F

try:  # scikit-image is used for the SLIC leaf segmentation
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
        segmentation: str = "slic",       # "slic" (default) or "grid"
        n_segments: int = 144,            # target leaf count (SLIC) or ~grid cells
        compactness: float = 2,           # SLIC compactness
        grid: tuple = (12, 12),           # grid leaves when segmentation="grid"
        value_mode: str = "prob",        # "logit" (paper Def 3) or "prob"
        # --- LIME prior ------------------------------------------------- #
        k_lime: int = 10,                  # number of top-LIME leaves used to seed S
        full_lime: bool = False,          # if True: ENTIRE chain follows LIME rank,
                                          # greedy phase skipped (ablation/control)
        # --- greedy selection signal ------------------------------------ #
        select_mode: str = "insertion",   # "insertion" (default, original v(S) rule)
                                          # or "both" (weighted insertion+deletion)
        both_alpha: float = 0.5,          # weight on insertion in "both" mode;
                                          # selection = a*v_ins + (1-a)*v_del
        lime_n_samples: int = 1000,       # LIME perturbation samples
        lime_kernel_width: float = 0.25,  # LIME locality kernel width
        lime_alpha: float = 1.0,          # LIME ridge regularization
        # --- audits / controls ----------------------------------------- #
        swap_audit: bool = True,          # run 1-swap local-optimality certificate
        swap_levels: Optional[list] = None,  # which k to audit; None -> a few checkpoints
        run_controls: bool = True,        # random + color + lime-order controls
        n_random_trees: int = 5,          # random-chain controls to average
        suff_eps: float = 0.05,           # sufficiency: v(S) >= (1-eps) v(root)
        # --- batching --------------------------------------------------- #
        max_batch: int = 64,              # max images per model forward pass
        seed: int = 0,
        **kw,
    ):
        super().__init__(*args, **kw)
        if value_mode not in ("logit", "prob"):
            raise ValueError(f"value_mode must be 'logit' or 'prob', got {value_mode!r}")
        if segmentation not in ("slic", "grid"):
            raise ValueError(f"segmentation must be 'slic' or 'grid', got {segmentation!r}")
        self.sigma = sigma
        self.segmentation = segmentation
        self.n_segments = n_segments
        self.compactness = compactness
        self.grid = grid
        self.value_mode = value_mode
        self.k_lime = int(k_lime)
        self.full_lime = bool(full_lime)
        if select_mode not in ("insertion", "both"):
            raise ValueError(f"select_mode must be 'insertion' or 'both', got {select_mode!r}")
        self.select_mode = select_mode
        self.both_alpha = float(both_alpha)
        self.lime_n_samples = lime_n_samples
        self.lime_kernel_width = lime_kernel_width
        self.lime_alpha = lime_alpha
        self.swap_audit = swap_audit
        self.swap_levels = swap_levels
        self.run_controls = run_controls
        self.n_random_trees = n_random_trees
        self.suff_eps = suff_eps
        self.max_batch = int(max_batch)
        self.seed = seed

    # ------------------------------------------------------------------ #
    # Phi: blur-completion projection operator (single image)
    # ------------------------------------------------------------------ #
    def _phi(self, x: torch.Tensor, b: torch.Tensor, mask_bool: np.ndarray) -> torch.Tensor:
        """Phi_R(x) = m * x + (1 - m) * b, m the (H,W) {0,1} region indicator."""
        m = torch.as_tensor(mask_bool, dtype=x.dtype, device=x.device)
        m = m.view(1, 1, *mask_bool.shape)  # (1,1,H,W) broadcast over channels
        return m * x + (1.0 - m) * b

    def _score(self, comp: torch.Tensor, target: int) -> float:
        """Raw model response on a single completed image: logit or prob."""
        with torch.no_grad():
            out = self.model(comp)
            if self.value_mode == "logit":
                return float(out[:, target].item())
            return float(F.softmax(out, dim=1)[:, target].item())

    # ------------------------------------------------------------------ #
    # BATCHED scoring core
    # ------------------------------------------------------------------ #
    def _score_batch(self, comps: torch.Tensor, target: int) -> np.ndarray:
        """Score a batch of completed images. comps: (N,C,H,W) -> (N,) np.float64.

        Chunks by self.max_batch and does ONE GPU->CPU sync per chunk.
        """
        n = comps.shape[0]
        out_scores = np.empty(n, dtype=np.float64)
        with torch.no_grad():
            for start in range(0, n, self.max_batch):
                cb = comps[start:start + self.max_batch]
                out = self.model(cb)
                if self.value_mode == "logit":
                    s = out[:, target]
                else:
                    s = F.softmax(out, dim=1)[:, target]
                out_scores[start:start + cb.shape[0]] = s.detach().cpu().numpy()
        return out_scores

    def _values_of_masks(
        self, masks_bool, x: torch.Tensor, b: torch.Tensor, x0_val: float, target: int
    ) -> np.ndarray:
        """v(R_i) for a list/array of (H,W) bool masks, evaluated in batches.

        Returns np.ndarray of shape (len(masks_bool),): score(Phi_{R_i}(x)) - x0_val.
        Builds completed images in chunks so peak memory stays at max_batch images.
        """
        if len(masks_bool) == 0:
            return np.empty(0, dtype=np.float64)
        marr = np.stack([np.asarray(m, dtype=bool) for m in masks_bool])  # (N,H,W)
        n = marr.shape[0]
        out_scores = np.empty(n, dtype=np.float64)
        with torch.no_grad():
            for start in range(0, n, self.max_batch):
                chunk = marr[start:start + self.max_batch]
                m = torch.as_tensor(chunk, dtype=x.dtype, device=x.device)  # (b,H,W)
                m = m.unsqueeze(1)                                          # (b,1,H,W)
                comps = m * x + (1.0 - m) * b                              # (b,C,H,W)
                out = self.model(comps)
                if self.value_mode == "logit":
                    s = out[:, target]
                else:
                    s = F.softmax(out, dim=1)[:, target]
                out_scores[start:start + chunk.shape[0]] = s.detach().cpu().numpy()
        return out_scores - x0_val

    def _value_of_mask(self, mask_bool, x, b, x0_val: float, target: int) -> float:
        """Single-mask convenience wrapper around the batched path."""
        return float(self._values_of_masks([mask_bool], x, b, x0_val, target)[0])

    def _deletion_drops(
        self, masks_bool, x: torch.Tensor, b: torch.Tensor, f_x: float, target: int
    ) -> np.ndarray:
        """Deletion-drop score for each (H,W) bool mask S (the IMPORTANT/kept set).

        Deletion removes S from the full sharp image: pixels IN S take the removal
        field, pixels OUTSIDE S stay sharp. The score is how far the target prob
        falls from the full-image value:

            del_drop(S) = f(x) - f( S->removal_field , complement->sharp )

        Same removal field as insertion (the blur reference `b`), per the "both"
        construction choice. Larger S removes more, so del_drop grows monotonically
        with S toward ~f(x) - f(x0) -- same scale and direction as v_ins, so a
        weighted sum a*v_ins + (1-a)*del_drop is dimensionally clean.

        NOTE: this is the deletion DROP (bigger = the set is more necessary). It is
        used ONLY as a selection signal in `both` mode; it is never stored as a
        node value, so Theorem 1 (which telescopes the insertion v) is untouched.
        """
        if len(masks_bool) == 0:
            return np.empty(0, dtype=np.float64)
        marr = np.stack([np.asarray(m, dtype=bool) for m in masks_bool])  # (N,H,W)
        n = marr.shape[0]
        out_scores = np.empty(n, dtype=np.float64)
        with torch.no_grad():
            for start in range(0, n, self.max_batch):
                chunk = marr[start:start + self.max_batch]
                m = torch.as_tensor(chunk, dtype=x.dtype, device=x.device)  # (b,H,W)
                m = m.unsqueeze(1)                                          # (b,1,H,W)
                # IN S -> removal field b ; OUTSIDE S -> sharp x   (opposite of Phi)
                comps = (1.0 - m) * x + m * b                              # (b,C,H,W)
                out = self.model(comps)
                if self.value_mode == "logit":
                    s = out[:, target]
                else:
                    s = F.softmax(out, dim=1)[:, target]
                out_scores[start:start + chunk.shape[0]] = s.detach().cpu().numpy()
        return f_x - out_scores

    # ------------------------------------------------------------------ #
    # leaf segmentation: SLIC (default) or fixed grid
    # ------------------------------------------------------------------ #
    def _leaf_labels(self, img01: np.ndarray) -> np.ndarray:
        """Return an (H,W) int label map of leaf regions."""
        H, W = img01.shape[:2]
        if self.segmentation == "grid":
            return self._grid_labels(H, W)
        if _HAS_SKIMAGE:
            labels = slic(
                img01,
                n_segments=self.n_segments,
                compactness=self.compactness,
                start_label=0,
                channel_axis=2,
            )
            return labels.astype(np.int64)
        return self._grid_labels(H, W)

    def _grid_labels(self, H: int, W: int) -> np.ndarray:
        """Regular grid tiling -> (H,W) int label map."""
        if self.segmentation == "grid":
            gh, gw = self.grid
        else:
            gh = gw = max(1, int(round(np.sqrt(self.n_segments))))
        ys = (np.arange(H) * gh // H).clip(max=gh - 1).astype(np.int64)
        xs = (np.arange(W) * gw // W).clip(max=gw - 1).astype(np.int64)
        labels = (ys[:, None] * gw + xs[None, :]).astype(np.int64)
        return labels

    # ------------------------------------------------------------------ #
    # LIME prior over leaves: weighted-ridge surrogate -> per-leaf coeff
    # (already batched in the original; kept, with max_batch as the chunk)
    # ------------------------------------------------------------------ #
    def _lime_rank(self, labels: np.ndarray, x, b, target: int):
        """Fit a LIME-style weighted linear surrogate on the leaf on/off vectors
        and return leaves ranked by descending surrogate coefficient.

        Returns (ranked_leaf_labels, coeff_by_label, n_forward_queries).
        """
        uniq = [int(l) for l in np.unique(labels)]
        n_cells = len(uniq)
        label_to_col = {l: j for j, l in enumerate(uniq)}

        col_map = np.empty_like(labels)
        for l in uniq:
            col_map[labels == l] = label_to_col[l]
        col_map_t = torch.as_tensor(col_map, dtype=torch.long, device=x.device)

        g = torch.Generator(device="cpu").manual_seed(self.seed)
        Z = (torch.rand(self.lime_n_samples, n_cells, generator=g) > 0.5).float()
        Z[0] = 1.0  # include the all-on sample (full image)

        probs = np.zeros(self.lime_n_samples, dtype=np.float64)
        n_q = 0
        with torch.no_grad():
            for start in range(0, self.lime_n_samples, self.max_batch):
                zb = Z[start:start + self.max_batch].to(x.device)   # (B,n_cells)
                keep = zb[:, col_map_t]                             # (B,H,W)
                keep = keep.unsqueeze(1)                            # (B,1,H,W)
                comp = keep * x + (1.0 - keep) * b
                out = self.model(comp)
                if self.value_mode == "logit":
                    p = out[:, target]
                else:
                    p = F.softmax(out, dim=1)[:, target]
                probs[start:start + zb.shape[0]] = p.detach().cpu().numpy()
                n_q += zb.shape[0]

        Znp = Z.cpu().numpy()
        all_on = np.ones(n_cells)
        d = 1.0 - (Znp @ all_on) / (
            np.linalg.norm(Znp, axis=1) * np.linalg.norm(all_on) + 1e-12
        )
        weights = np.exp(-(d ** 2) / (self.lime_kernel_width ** 2))

        coefs = _weighted_ridge(Znp, probs, weights, alpha=self.lime_alpha)

        coeff_by_label = {l: float(coefs[label_to_col[l]]) for l in uniq}
        ranked = sorted(uniq, key=lambda l: coeff_by_label[l], reverse=True)
        return ranked, coeff_by_label, n_q

    # ------------------------------------------------------------------ #
    # LIME-seeded nested greedy-add-leaf chain on the joint score v(union)
    # ------------------------------------------------------------------ #
    def _build_tree(self, img01, labels, x, b, x0_val: float, target: int, f_x: float):
        """Grow one nested coalition: seed with the top-`k_lime` LIME leaves (in
        LIME order), then greedily add the leaf that most increases v(S u {l}).

        Greedy phase is batched: each step scores ALL remaining candidates in one
        batched forward and picks the argmax. If `self.full_lime` is True the seed
        covers EVERY leaf, so the greedy phase never runs.
        Returns (root, info).
        """
        uniq = [int(l) for l in np.unique(labels)]
        leaf_mask = {l: (labels == l) for l in uniq}
        leaf_color = {l: img01[leaf_mask[l]].mean(axis=0) for l in uniq}

        n_fresh_q = 0

        # ---- LIME prior ranking --------------------------------------- #
        lime_ranked, lime_coeff, n_lime_q = self._lime_rank(labels, x, b, target)

        # ---- leaf nodes + individual values (ONE batch over all leaves) #
        leaf_value_arr = self._values_of_masks(
            [leaf_mask[l] for l in uniq], x, b, x0_val, target
        )
        n_fresh_q += len(uniq)

        node: dict[int, _Node] = {}
        leaf_node: dict[int, _Node] = {}
        leaf_value: dict[int, float] = {}
        next_id = 0
        for l, v_l in zip(uniq, leaf_value_arr):
            v_l = float(v_l)
            nd = _Node(id=next_id, mask=leaf_mask[l], children=[], v=v_l)
            node[next_id] = nd
            leaf_node[l] = nd
            leaf_value[l] = v_l
            next_id += 1

        # ---- chain bookkeeping ---------------------------------------- #
        chain = []
        chain_v = []
        chain_delta = []
        chain_node_ids = []
        chain_source = []
        merge_order = []
        in_set: set = set()

        # ---- seed phase: top-k LIME leaves, in LIME rank order -------- #
        if self.full_lime:
            k_seed = len(uniq)
            seed_source = "lime_full"
        else:
            k_seed = max(1, min(self.k_lime, len(uniq)))
            seed_source = "lime_seed"
        seed_leaves = lime_ranked[:k_seed]

        cur_node = None
        cur_mask = None
        cur_v = None
        for l in seed_leaves:
            if cur_node is None:
                cur_mask = leaf_mask[l].copy()
                cur_v = leaf_value[l]
                cur_node = leaf_node[l]
                in_set.add(l)
                chain.append(l)
                chain_v.append(cur_v)
                chain_delta.append(0.0)
                chain_node_ids.append(cur_node.id)
                chain_source.append(seed_source)
                continue
            union_mask = cur_mask | leaf_mask[l]
            v_union = self._value_of_mask(union_mask, x, b, x0_val, target)
            n_fresh_q += 1
            leaf_child = leaf_node[l]
            merged = _Node(
                id=next_id,
                mask=union_mask,
                children=[cur_node, leaf_child],
                v=v_union,
            )
            merged.delta = v_union - (cur_v + leaf_value[l])
            node[next_id] = merged
            merge_order.append((cur_node.id, leaf_child.id, next_id, float(merged.delta)))
            cur_node, cur_mask, cur_v = merged, union_mask, v_union
            in_set.add(l)
            next_id += 1
            chain.append(l)
            chain_v.append(cur_v)
            chain_delta.append(float(merged.delta))
            chain_node_ids.append(cur_node.id)
            chain_source.append(seed_source)

        # ---- greedy phase: batched full-evaluation per step ----------- #
        # Skipped entirely in full_lime mode (in_set already == all leaves).
        # select_mode controls WHICH leaf is picked each step:
        #   "insertion": argmax v_ins(S u {l})            (original rule)
        #   "both"     : argmax [ a*v_ins + (1-a)*del_drop ]
        # In BOTH cases the node value stored is the insertion v(S u {l}), so the
        # telescoping decomposition (Theorem 1) is identical across modes; only
        # the leaf ORDER differs.
        if not self.full_lime:
            a = self.both_alpha
            while len(in_set) < len(uniq):
                remaining = [l for l in uniq if l not in in_set]
                # build one candidate union mask per remaining leaf, score in batch
                cand_masks = [cur_mask | leaf_mask[l] for l in remaining]
                cand_v = self._values_of_masks(cand_masks, x, b, x0_val, target)
                n_fresh_q += len(remaining)

                if self.select_mode == "both":
                    cand_del = self._deletion_drops(cand_masks, x, b, f_x, target)
                    n_fresh_q += len(remaining)
                    select_score = a * cand_v + (1.0 - a) * cand_del
                else:
                    select_score = cand_v

                best_idx = int(np.argmax(select_score))  # argmax of selection signal
                l = remaining[best_idx]
                best_union_mask = cand_masks[best_idx]
                best_union_v = float(cand_v[best_idx])    # stored value is ALWAYS v_ins

                leaf_child = leaf_node[l]
                merged = _Node(
                    id=next_id,
                    mask=best_union_mask,
                    children=[cur_node, leaf_child],
                    v=best_union_v,
                )
                merged.delta = best_union_v - (cur_v + leaf_value[l])
                node[next_id] = merged
                merge_order.append((cur_node.id, leaf_child.id, next_id, float(merged.delta)))

                cur_node, cur_mask, cur_v = merged, best_union_mask, best_union_v
                in_set.add(l)
                next_id += 1

                chain.append(l)
                chain_v.append(cur_v)
                chain_delta.append(float(merged.delta))
                chain_node_ids.append(cur_node.id)
                chain_source.append("greedy")

        root = cur_node

        # ---- sufficiency node: smallest-area node with v >= (1-eps) v_root #
        v_root_est = root.v
        thresh = (1.0 - self.suff_eps) * v_root_est if v_root_est > 0 else v_root_est
        suff_node, suff_area = root, root.area
        for nd in node.values():
            if nd.v >= thresh and nd.area < suff_area:
                suff_node, suff_area = nd, nd.area

        info = {
            "chain": chain,
            "chain_v": chain_v,
            "chain_delta": chain_delta,
            "chain_node_ids": chain_node_ids,
            "chain_source": chain_source,
            "merge_order": merge_order,
            "v_root_est": float(v_root_est),
            "suff_node_id": int(suff_node.id),
            "suff_v": float(suff_node.v),
            "n_construction_queries": int(n_fresh_q),
            "n_lime_queries": int(n_lime_q),
            "k_lime_used": int(k_seed),
            "full_lime": bool(self.full_lime),
            "lime_ranked": [int(l) for l in lime_ranked],
            "lime_seed_leaves": [int(l) for l in seed_leaves],
            "lime_coeff": {int(l): float(c) for l, c in lime_coeff.items()},
            "leaf_value": leaf_value,
            "leaf_mask": leaf_mask,
            "leaf_color": leaf_color,
            "uniq": uniq,
        }
        return root, info

    # ------------------------------------------------------------------ #
    # 1-swap local-optimality audit (BATCHED over all in/out pairs)
    # ------------------------------------------------------------------ #
    def _swap_audit(self, tinfo, x, b, x0_val, target):
        """For selected levels k, test whether any single in/out swap raises
        v(S_k). All pairs for a level are scored in one batched call. Masks are
        built by toggling the two swapped leaves on a precomputed base mask.
        Returns per-level certificates and counts fresh forwards used."""
        chain = tinfo["chain"]
        leaf_mask = tinfo["leaf_mask"]
        uniq = tinfo["uniq"]
        n = len(uniq)

        if self.swap_levels is not None:
            levels = [k for k in self.swap_levels if 1 <= k <= len(chain)]
        else:
            cand = sorted({1, max(1, n // 8), max(1, n // 4), max(1, n // 2)})
            levels = [k for k in cand if k <= len(chain)]

        n_q = 0
        audits = {}
        any_mask = next(iter(leaf_mask.values()))
        for k in levels:
            S = set(chain[:k])
            outside = [l for l in uniq if l not in S]

            # base mask = union of S
            base_mask = np.zeros_like(any_mask)
            for l in S:
                base_mask = base_mask | leaf_mask[l]
            v_S = self._value_of_mask(base_mask, x, b, x0_val, target)
            n_q += 1

            # enumerate all (l_in, l_out) pairs; build masks by toggling 2 leaves
            pairs = []
            swap_masks = []
            for l_in in S:
                # removing l_in from the union: base minus l_in's pixels
                mask_wo_in = base_mask & ~leaf_mask[l_in]
                for l_out in outside:
                    swap_masks.append(mask_wo_in | leaf_mask[l_out])
                    pairs.append((int(l_in), int(l_out)))

            if swap_masks:
                v_sw = self._values_of_masks(swap_masks, x, b, x0_val, target)
                n_q += len(swap_masks)
                improve = v_sw - v_S
                j = int(np.argmax(improve))
                best_improve = float(improve[j])
                best_swap = pairs[j] if best_improve > 0.0 else None
            else:
                best_improve = 0.0
                best_swap = None

            audits[int(k)] = {
                "v_S": float(v_S),
                "is_1swap_local_opt": bool(best_swap is None),
                "best_improving_swap": best_swap,
                "best_improve": float(best_improve),
            }
        return audits, n_q

    # ------------------------------------------------------------------ #
    # control chains (each nested chain scored as ONE batched mask set)
    # ------------------------------------------------------------------ #
    def _control_curves(self, tinfo, x, b, x0_val, target):
        """Random-leaf-order, color-order, and LIME-order nested chains. Each
        chain's cumulative masks are precomputed and scored in a single batched
        call. Returns mean curves and forward count."""
        uniq = tinfo["uniq"]
        leaf_mask = tinfo["leaf_mask"]
        leaf_color = tinfo["leaf_color"]
        lime_ranked = tinfo["lime_ranked"]
        rng = np.random.default_rng(self.seed)
        n_q = 0
        any_mask = next(iter(leaf_mask.values()))

        def curve_for_order(order):
            nonlocal n_q
            # precompute the n nested cumulative masks, then score them all at once
            cum = np.zeros_like(any_mask)
            masks = []
            for l in order:
                cum = cum | leaf_mask[l]
                masks.append(cum.copy())
            vs = self._values_of_masks(masks, x, b, x0_val, target)
            n_q += len(masks)
            return vs.tolist()

        # random controls (averaged)
        rand_curves = []
        for _ in range(self.n_random_trees):
            order = list(uniq)
            rng.shuffle(order)
            rand_curves.append(curve_for_order(order))
        rand_mean = np.mean(np.array(rand_curves), axis=0).tolist()

        # color control: descending brightness (model-blind heuristic)
        bright = {l: float(np.mean(leaf_color[l])) for l in uniq}
        color_order = sorted(uniq, key=lambda l: bright[l], reverse=True)
        color_curve = curve_for_order(color_order)

        # LIME-order control: pure surrogate rank, no greedy v(S) decisions.
        lime_curve = curve_for_order(list(lime_ranked))

        return {
            "random_mean_curve": rand_mean,
            "color_curve": color_curve,
            "lime_curve": lime_curve,
            "n_random_trees": self.n_random_trees,
        }, n_q

    # ------------------------------------------------------------------ #
    # tree traversal
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
        root, tinfo = self._build_tree(img01, labels, x, b, x0_val, target, f_x)

        leaves, internals, _ = self._collect(root)
        leaf_masks = {leaf.id: leaf.mask for leaf in leaves}
        leaf_masks_by_label = {int(l): m for l, m in tinfo["leaf_mask"].items()}

        # --- telescoping identity (Delta set at construction) --------------- #
        sum_leaf_v = float(sum(l.v for l in leaves))
        sum_delta = float(sum(r.delta for r in internals))
        identity_lhs = float(root.v)
        identity_rhs = sum_leaf_v + sum_delta
        identity_residual = identity_lhs - identity_rhs

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
        n_lime_q = tinfo["n_lime_queries"]
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
                "segmentation": self.segmentation,
                "n_segments": self.n_segments,
                "grid": self.grid,
                "n_leaves": len(leaves),
                "n_internal": len(internals),
                "merge_rule": ("lime_full_order" if self.full_lime
                               else "lime_seeded_nested_greedy_joint_score"),
                "select_mode": self.select_mode,
                "both_alpha": self.both_alpha,
                # LIME prior
                "k_lime": self.k_lime,
                "k_lime_used": tinfo["k_lime_used"],
                "full_lime": tinfo["full_lime"],
                "lime_ranked": tinfo["lime_ranked"],
                "lime_seed_leaves": tinfo["lime_seed_leaves"],
                "lime_coeff": tinfo["lime_coeff"],
                "chain_source": tinfo["chain_source"],
                # query accounting
                "n_lime_queries": n_lime_q,
                "n_construction_queries": n_construction_q,
                "n_audit_queries": n_audit_q,
                "n_control_queries": n_control_q,
                "n_total_queries": n_lime_q + n_construction_q + n_audit_q + n_control_q,
                "max_batch": self.max_batch,
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
                "chain_v": tinfo["chain_v"],
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
                "swap_audits": swap_audits,
                "controls": controls,
                # full tree + leaf masks
                "tree": all_serialized,
                "leaf_masks": leaf_masks,
                "leaf_masks_by_label": leaf_masks_by_label,
                "leaf_labels": labels,
                "reference": "blur_completion",
            },
        )


def _weighted_ridge(Z, y, w, alpha=1.0):
    """Closed-form weighted ridge; returns per-feature coefficients (intercept dropped)."""
    n, d = Z.shape
    Zb = np.concatenate([Z, np.ones((n, 1))], axis=1)  # add intercept
    Wd = w[:, None]
    A = Zb.T @ (Wd * Zb)
    reg = alpha * np.eye(d + 1)
    reg[-1, -1] = 0.0  # don't regularize intercept
    A += reg
    rhs = Zb.T @ (w * y)
    sol = np.linalg.solve(A, rhs)
    return sol[:-1]  # drop intercept